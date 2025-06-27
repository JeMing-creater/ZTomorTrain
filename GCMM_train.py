import os

# os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import sys
from datetime import datetime
from typing import Dict

import monai
import torch
import yaml
from tqdm import tqdm
import torch.nn as nn
from dataclasses import dataclass, field
from accelerate import Accelerator
from easydict import EasyDict
from monai.utils import ensure_tuple_rep
from objprint import objstr
from timm.optim import optim_factory
from accelerate.utils import DistributedDataParallelKwargs
from termcolor import colored

from src import utils
from get_model import get_model
from src.loader import get_dataloader_GCNC as get_dataloader
from src.optimizer import LinearWarmupCosineAnnealingLR
from src.utils import Logger, write_example, resume_train_state, split_metrics

from src.model.HWAUNETR_class import HWAUNETR
from src.model.FMUNETR_class_seg import FMUNETR
from src.model.SwinUNETR import MultiTaskSwinUNETR
from monai.networks.nets import SwinUNETR


def freeze_seg_decoder(model):
    """
    冻结 Seg_Decoder 模块的所有参数，适配 accelerate 多卡训练
    """
    for name, param in model.named_parameters():
        if "Seg_Decoder" in name:
            param.requires_grad = False  # 停止梯度更新
            if param.grad is not None:
                param.grad.detach_()  # 清理梯度，防止错误同步

    # 强制设置 eval 模式，防止 BN、Dropout 引发 DDP 不一致
    if hasattr(model, "Seg_Decoder"):
        model.Seg_Decoder.eval()


def freeze_class_model(model):
    """
    在每轮 train_one_epoch 前调用，用于动态冻结 Seg_Decoder
    """

    for name, param in model.named_parameters():
        if "Encoder" in name or "Class_Decoder" in name:
            param.requires_grad = False  # 停止梯度更新
            if param.grad is not None:
                param.grad.detach_()  # 清理梯度，防止错误同步

    # 强制设置 eval 模式，防止 BN、Dropout 引发 DDP 不一致
    if hasattr(model, "Encoder"):
        model.Encoder.eval()

    if hasattr(model, "Class_Decoder"):
        model.Class_Decoder.eval()


def train_class_one_epoch(
    model: torch.nn.Module,
    loss_params: Dict[str, torch.nn.Parameter],
    loss_functions: Dict[str, torch.nn.modules.loss._Loss],
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    metrics: Dict[str, monai.metrics.CumulativeIterationMetric],
    post_trans: monai.transforms.Compose,
    accelerator: Accelerator,
    epoch: int,
    train_step: int,
):
    # 训练
    model.train()
    freeze_seg_decoder(model)
    accelerator.print(f"Training...", flush=True)
    loop = tqdm(enumerate(train_loader), total=len(train_loader))
    # for i, image_batch in enumerate(train_loader):
    for i, image_batch in loop:
        # get model output.
        CLS_out, _ = model(image_batch["image"])
        pdl1_y = CLS_out[0]
        m_y = CLS_out[1]
        # get label.
        pdl1_label = image_batch["pdl1_label"]
        m_label = image_batch["m_label"]

        # get loss.
        total_loss = 0
        for name in loss_functions:
            if name == "dice_loss":
                continue
            loss1 = loss_functions[name](pdl1_y, pdl1_label.float())
            loss2 = loss_functions[name](m_y, m_label.float())
            loss = loss_params["pdl1"] * loss1 + loss_params["m"] * loss2
            accelerator.log({"Train Class/" + name: float(loss)}, step=train_step)
            total_loss += loss

        # get metric compute.
        pdl1_y_pred = post_trans(pdl1_y)
        m_y_pred = post_trans(m_y)

        # PD L1 metrics compute.
        for metric_name in metrics["PD_L1"]:
            if metric_name == "miou_metric":
                pdl1_y_pred = pdl1_y_pred.unsqueeze(2)
                pdl1_label = pdl1_label.unsqueeze(2)
            elif metric_name == "dice_metric" or metric_name == "hd95_metric":
                continue
            else:
                pdl1_y_pred = pdl1_y_pred
                pdl1_label = pdl1_label
            metrics["PD_L1"][metric_name](y_pred=pdl1_y_pred, y=pdl1_label)

        # M metrics compute.
        for metric_name in metrics["M"]:
            if metric_name == "miou_metric":
                m_y_pred = m_y_pred.unsqueeze(2)
                m_label = m_label.unsqueeze(2)
            elif metric_name == "dice_metric" or metric_name == "hd95_metric":
                continue
            else:
                m_y_pred = m_y_pred
                m_label = m_label
            metrics["M"][metric_name](y_pred=m_y_pred, y=m_label)

        # loss backward.
        accelerator.backward(total_loss)
        optimizer.step()
        optimizer.zero_grad()

        # # chack which param not be used to training.
        # for name, param in model.named_parameters():
        #     if param.grad is None:
        #         print(name)

        # log writed.
        accelerator.log(
            {
                "Train/Total Loss": float(total_loss),
            },
            step=train_step,
        )
        # 更新信息
        loop.set_description(
            colored(f"Epoch [{epoch+1}", "red") + f"/{config.trainer.num_epochs}]"
        )
        loop.set_postfix(loss=total_loss)

        train_step += 1

    scheduler.step(epoch)

    # compute all metric data.
    metric = {}
    for label_name in metrics:
        if label_name == "Seg":
            continue
        for metric_name in metrics[label_name]:
            batch_acc = (
                metrics[label_name][metric_name].aggregate()[0].to(accelerator.device)
            )

            if accelerator.num_processes > 1:
                batch_acc = accelerator.reduce(batch_acc) / accelerator.num_processes

            # give every single task metric
            metrics[label_name][metric_name].reset()
            metric.update(
                {
                    f"Train {label_name}/{metric_name}": float(batch_acc.mean()),
                }
            )

    for label_name in metrics:
        if label_name == "Seg":
            continue
        for metric_name in metrics[label_name]:
            all_data = []
            for key in metric.keys():
                if metric_name in key and label_name in key:
                    all_data.append(metric[key])
            me_data = sum(all_data) / len(all_data)
            metric.update({f"Train {label_name}/{metric_name}": float(me_data)})

    accelerator.log(metric, step=epoch)
    return metric, train_step


def train_seg_one_epoch(
    model: torch.nn.Module,
    loss_params: Dict[str, torch.nn.Parameter],
    loss_functions: Dict[str, torch.nn.modules.loss._Loss],
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    metrics: Dict[str, monai.metrics.CumulativeIterationMetric],
    post_trans: monai.transforms.Compose,
    accelerator: Accelerator,
    epoch: int,
    train_step: int,
):
    # 训练
    model.train()
    freeze_class_model(model)
    accelerator.print(f"Training...", flush=True)
    loop = tqdm(enumerate(train_loader), total=len(train_loader))
    # for i, image_batch in enumerate(train_loader):
    for i, image_batch in loop:
        # get model output.
        _, SEG_out = model(image_batch["image"])

        # get label.
        seg_label = image_batch["label"]

        # get loss.
        total_loss = 0
        for name in loss_functions:
            if name != "dice_loss":
                continue
            else:
                loss = loss_functions[name](SEG_out, seg_label.float())
            accelerator.log({"Train/" + name: float(loss)}, step=train_step)
            total_loss += loss

        # get metric compute.
        seg_pred = post_trans(SEG_out)

        # Seg metrics compute.
        for metric_name in metrics["Seg"]:
            metrics["Seg"][metric_name](y_pred=seg_pred, y=seg_label)

        # log writed.
        accelerator.log(
            {
                f"Train/Total Loss": float(total_loss),
            },
            step=train_step,
        )

        # 更新信息
        loop.set_description(
            colored(f"Epoch [{epoch+1}", "red") + f"/{config.trainer.num_epochs}]"
        )
        loop.set_postfix(loss=total_loss)

        # loss backward.
        accelerator.backward(total_loss)
        optimizer.step()
        optimizer.zero_grad()

        # # chack which param not be used to training.
        # for name, param in model.named_parameters():
        #     if param.grad is None:
        #         print(name)

        # log writed.
        accelerator.log(
            {
                "Train/Total Loss": float(total_loss),
            },
            step=train_step,
        )
        # 更新信息
        loop.set_description(f"Epoch [{epoch+1}/{config.trainer.num_epochs}]")
        loop.set_postfix(loss=total_loss)

        train_step += 1

    scheduler.step(epoch)

    # compute all metric data.
    metric = {}
    for label_name in metrics:
        if label_name == "Seg":
            for metric_name in metrics[label_name]:
                batch_acc = (
                    metrics[label_name][metric_name]
                    .aggregate()[0]
                    .to(accelerator.device)
                )

                if accelerator.num_processes > 1:
                    batch_acc = (
                        accelerator.reduce(batch_acc) / accelerator.num_processes
                    )

                # give every single task metric
                metrics[label_name][metric_name].reset()

                metric_dice = {}
                metric_dice[f"Train/mean {metric_name}"] = float(batch_acc.mean())

                for i in range(len(config.GCNC_loader.checkModels)):
                    metric_dice[f"Train/{config.GCNC_loader.checkModels[i]}"] = float(
                        batch_acc[i]
                    )

                metric.update(metric_dice)

    accelerator.log(metric, step=epoch)
    return metric, train_step


@torch.no_grad()
def val_one_epoch(
    model: torch.nn.Module,
    loss_params: Dict[str, torch.nn.Parameter],
    loss_functions: Dict[str, torch.nn.modules.loss._Loss],
    val_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    metrics: Dict[str, monai.metrics.CumulativeIterationMetric],
    post_trans: monai.transforms.Compose,
    accelerator: Accelerator,
    epoch: int,
    val_step: int,
    test: bool = False,
):
    # 训练
    model.eval()
    if test:
        flag = "Test"
        accelerator.print(f"Testing...", flush=True)
    else:
        flag = "Val"
        accelerator.print(f"Valing...", flush=True)
    loop = tqdm(enumerate(val_loader), total=len(val_loader))
    # for i, image_batch in enumerate(train_loader):
    for i, image_batch in loop:
        # get model output.
        CLS_out, Seg_Out = model(image_batch["image"])
        pdl1_y = CLS_out[0]
        m_y = CLS_out[1]
        # get label.
        pdl1_label = image_batch["pdl1_label"]
        m_label = image_batch["m_label"]
        seg_label = image_batch["label"]

        # get loss.
        total_loss = 0
        for name in loss_functions:
            if name == "dice_loss":
                loss = loss_functions[name](Seg_Out, seg_label.float())
            else:
                loss1 = loss_functions[name](pdl1_y, pdl1_label.float())
                loss2 = loss_functions[name](m_y, m_label.float())
                loss = loss_params["pdl1"] * loss1 + loss_params["m"] * loss2
            accelerator.log({f"{flag} Class/" + name: float(loss)}, step=val_step)
            total_loss += loss

        # get metric compute.
        pdl1_y_pred = post_trans(pdl1_y)
        m_y_pred = post_trans(m_y)
        seg_pred = post_trans(Seg_Out)

        # PD L1 metrics compute.
        for metric_name in metrics["PD_L1"]:
            if metric_name == "miou_metric":
                pdl1_y_pred = pdl1_y_pred.unsqueeze(2)
                pdl1_label = pdl1_label.unsqueeze(2)
            elif metric_name == "dice_metric" or metric_name == "hd95_metric":
                continue
            else:
                pdl1_y_pred = pdl1_y_pred
                pdl1_label = pdl1_label
            metrics["PD_L1"][metric_name](y_pred=pdl1_y_pred, y=pdl1_label)

        # M metrics compute.
        for metric_name in metrics["M"]:
            if metric_name == "miou_metric":
                m_y_pred = m_y_pred.unsqueeze(2)
                m_label = m_label.unsqueeze(2)
            elif metric_name == "dice_metric" or metric_name == "hd95_metric":
                continue
            else:
                m_y_pred = m_y_pred
                m_label = m_label
            metrics["M"][metric_name](y_pred=m_y_pred, y=m_label)

        # Seg metrics compute.
        for metric_name in metrics["Seg"]:
            metrics["Seg"][metric_name](y_pred=seg_pred, y=seg_label)

        # log writed.
        accelerator.log(
            {
                f"{flag}/Total Loss": float(total_loss),
            },
            step=val_step,
        )
        # 更新信息
        loop.set_description(
            colored(f"Epoch [{epoch+1}", "green") + f"/{config.trainer.num_epochs}]"
        )
        loop.set_postfix(loss=total_loss)

        val_step += 1

    # compute all metric data.
    metric = {}
    for label_name in metrics:
        for metric_name in metrics[label_name]:
            batch_acc = (
                metrics[label_name][metric_name].aggregate()[0].to(accelerator.device)
            )

            if accelerator.num_processes > 1:
                batch_acc = accelerator.reduce(batch_acc) / accelerator.num_processes

            # give every single task metric
            metrics[label_name][metric_name].reset()
            if label_name == "Seg":
                metric_dice = {}
                metric_dice[f"{flag}/mean {metric_name}"] = float(batch_acc.mean())

                for i in range(len(config.GCNC_loader.checkModels)):
                    metric_dice[f"{flag}/{config.GCNC_loader.checkModels[i]}"] = float(
                        batch_acc[i]
                    )
                metric.update(metric_dice)
            else:
                metric.update(
                    {
                        f"{flag} {label_name}/{metric_name}": float(batch_acc.mean()),
                    }
                )

    for label_name in metrics:
        if label_name == "Seg":
            continue
        for metric_name in metrics[label_name]:
            all_data = []
            for key in metric.keys():
                if metric_name in key and label_name in key:
                    all_data.append(metric[key])
            me_data = sum(all_data) / len(all_data)
            metric.update({f"{flag} {label_name}/{metric_name}": float(me_data)})

    accelerator.log(metric, step=epoch)
    return metric, val_step


if __name__ == "__main__":
    config = EasyDict(
        yaml.load(open("config.yml", "r", encoding="utf-8"), Loader=yaml.FullLoader)
    )
    utils.same_seeds(50)
    logging_dir = (
        os.getcwd()
        + "/logs/"
        + config.finetune.GCNC.checkpoint
        + str(datetime.now())
        .replace(" ", "_")
        .replace("-", "_")
        .replace(":", "_")
        .replace(".", "_")
    )
    accelerator = Accelerator(
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
        cpu=False,
        log_with=["tensorboard"],
        project_dir=logging_dir,
    )
    Logger(logging_dir if accelerator.is_local_main_process else None)
    accelerator.init_trackers(os.path.split(__file__)[-1].split(".")[0])
    accelerator.print(objstr(config))

    accelerator.print("load model...")
    model = get_model(config)
    accelerator.print("load dataset...")
    train_loader, val_loader, test_loader, example = get_dataloader(config)

    # keep example log
    if accelerator.is_main_process == True:
        write_example(example, logging_dir)

    inference = monai.inferers.SlidingWindowInferer(
        roi_size=config.GCNC_loader.target_size,
        overlap=0.5,
        sw_device=accelerator.device,
        device=accelerator.device,
    )

    loss_params = {
        "pdl1": 1.0,
        "m": 0.0,
    }

    loss_functions = {
        "focal_loss": monai.losses.FocalLoss(to_onehot_y=False),
        "bce_loss": nn.BCEWithLogitsLoss().to(accelerator.device),
        "dice_loss": monai.losses.DiceLoss(
            smooth_nr=0, smooth_dr=1e-5, to_onehot_y=False, sigmoid=True
        ),
    }

    metrics = {
        "PD_L1": {
            "accuracy": monai.metrics.ConfusionMatrixMetric(
                include_background=False, metric_name="accuracy"
            ),
            "f1": monai.metrics.ConfusionMatrixMetric(
                include_background=False, metric_name="f1 score"
            ),
            "specificity": monai.metrics.ConfusionMatrixMetric(
                include_background=False, metric_name="specificity"
            ),
            "recall": monai.metrics.ConfusionMatrixMetric(
                include_background=False, metric_name="recall"
            ),
            "miou_metric": monai.metrics.MeanIoU(include_background=False),
        },
        "M": {
            "accuracy": monai.metrics.ConfusionMatrixMetric(
                include_background=False, metric_name="accuracy"
            ),
            "f1": monai.metrics.ConfusionMatrixMetric(
                include_background=False, metric_name="f1 score"
            ),
            "specificity": monai.metrics.ConfusionMatrixMetric(
                include_background=False, metric_name="specificity"
            ),
            "recall": monai.metrics.ConfusionMatrixMetric(
                include_background=False, metric_name="recall"
            ),
            "miou_metric": monai.metrics.MeanIoU(include_background=False),
        },
        "Seg": {
            "dice_metric": monai.metrics.DiceMetric(
                include_background=True,
                reduction=monai.utils.MetricReduction.MEAN_BATCH,
                get_not_nans=True,
            ),
            "hd95_metric": monai.metrics.HausdorffDistanceMetric(
                percentile=95,
                include_background=True,
                reduction=monai.utils.MetricReduction.MEAN_BATCH,
                get_not_nans=True,
            ),
        },
    }

    post_trans = monai.transforms.Compose(
        [
            monai.transforms.Activations(sigmoid=True),
            monai.transforms.AsDiscrete(threshold=0.5),
        ]
    )

    optimizer = optim_factory.create_optimizer_v2(
        model,
        opt=config.trainer.optimizer,
        weight_decay=float(config.trainer.weight_decay),
        lr=float(config.trainer.lr),
        betas=(config.trainer.betas[0], config.trainer.betas[1]),
    )

    # optimizer_seg = optim_factory.create_optimizer_v2(
    #     model,
    #     opt=config.trainer.optimizer,
    #     weight_decay=float(config.trainer.weight_decay),
    #     lr=float(config.trainer.lr),
    #     betas=(config.trainer.betas[0], config.trainer.betas[1]),
    # )

    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer,
        warmup_epochs=config.trainer.warmup,
        max_epochs=config.trainer.num_epochs,
    )

    # scheduler_seg = LinearWarmupCosineAnnealingLR(
    #     optimizer_seg,
    #     warmup_epochs=config.trainer.warmup,
    #     max_epochs=config.trainer.num_epochs,
    # )

    # start training
    accelerator.print("Start Training! \n")
    train_step = 0
    best_eopch = -1
    val_step = 0

    best_val_metric_data = torch.tensor(0)
    best_val_metrics = {}

    best_test_metric_data = torch.tensor(0)
    best_test_metrics = {}
    
    test_M = 0

    starting_epoch = 0

    if config.trainer.resume:
        (
            model,
            optimizer,
            scheduler,
            starting_epoch,
            train_step,
            best_val_metric_data,
            best_test_metric_data,
            best_val_metrics,
            best_test_metrics,
        ) = utils.resume_train_state(
            model,
            "{}".format(config.finetune.GCNC.checkpoint),
            optimizer,
            scheduler,
            train_loader,
            accelerator,
            seg=False,
        )
        val_step = train_step

    model, optimizer, scheduler, train_loader, val_loader, test_loader = (
        accelerator.prepare(
            model, optimizer, scheduler, train_loader, val_loader, test_loader
        )
    )

    best_val_metric_data = torch.Tensor([best_val_metric_data]).to(accelerator.device)
    best_test_metric_data = torch.Tensor([best_test_metric_data]).to(accelerator.device)

    for epoch in range(starting_epoch, config.trainer.num_epochs):
        if epoch == config.trainer.seg_epochs:
            # 当分类与分割任务转换，清空分类精度
            best_val_metric_data = torch.Tensor([0]).to(accelerator.device)
            best_test_metric_data = torch.Tensor([0]).to(accelerator.device)
        if epoch < config.trainer.seg_epochs:
            # 首先训练分类任务
            train_metric, train_step = train_class_one_epoch(
                model=model,
                loss_params=loss_params,
                loss_functions=loss_functions,
                train_loader=train_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                metrics=metrics,
                post_trans=post_trans,
                accelerator=accelerator,
                epoch=epoch,
                train_step=train_step,
            )
        else:
            # 然后训练分割任务
            train_metric, train_step = train_seg_one_epoch(
                model=model,
                loss_params=loss_params,
                loss_functions=loss_functions,
                train_loader=train_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                metrics=metrics,
                post_trans=post_trans,
                accelerator=accelerator,
                epoch=epoch,
                train_step=train_step,
            )

        val_metric, val_step = val_one_epoch(
            model=model,
            loss_params=loss_params,
            loss_functions=loss_functions,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            metrics=metrics,
            post_trans=post_trans,
            accelerator=accelerator,
            epoch=epoch,
            val_step=val_step,
            test=False,
        )

        if epoch < config.trainer.seg_epochs:

            val_top = val_metric["Val PD_L1/accuracy"]
            if best_val_metric_data < val_top:
                best_val_metric_data = torch.Tensor([val_top]).to(accelerator.device)
                best_val_metrics = val_metric
                best_eopch = epoch
                accelerator.print(
                    f"New Best Val PD_L1/accuracy: {val_top}, at epoch {epoch+1}\n"
                )

                accelerator.print(
                    colored(f"Epoch [{epoch+1}", "red")
                    + f"/{config.trainer.num_epochs}]  Now train PD L1 acc: {train_metric['Train PD_L1/accuracy']}, Now train M acc: {train_metric['Train M/accuracy']}\n"
                )

                if config.GCNC_loader.fusion != True:
                    test_metric, _ = val_one_epoch(
                        model=model,
                        loss_params=loss_params,
                        loss_functions=loss_functions,
                        train_loader=test_loader,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        metrics=metrics,
                        post_trans=post_trans,
                        accelerator=accelerator,
                        epoch=epoch,
                        val_step=0,
                        test=True,
                    )
                    best_test_metric_data = test_metric["Test PD_L1/accuracy"]
                    best_test_metrics = test_metric
                    test_M = test_metric["Test M/accuracy"]
                else:
                    test_metric = val_metric
                    best_test_metric_data = test_metric["Val PD_L1/accuracy"]
                    best_test_metrics = test_metric
                    test_M = test_metric["Val M/accuracy"]

                # 保存模型
                accelerator.save_state(
                    output_dir=f"{os.getcwd()}/model_store/{config.finetune.GCM.checkpoint}/best"
                )
            accelerator.print(
                colored(f"Epoch [{epoch+1}", "green")
                + f"/{config.trainer.num_epochs}]  Now val PD L1 acc: {val_top}, Now val M acc: {val_metric['Val M/accuracy']}, best test PD L1 acc: {best_val_metric_data}, best test M acc: {test_M}"
            )
        else:
            val_top = val_metric["Seg/mean dice_metric"]
            if best_val_metric_data < val_top:
                best_val_metric_data = torch.Tensor([val_top]).to(accelerator.device)
                best_val_metrics = val_metric
                best_eopch = epoch

                accelerator.print(
                    colored(f"Epoch [{epoch+1}", "red")
                    + f"/{config.trainer.num_epochs}]  Now train Seg dice: {train_metric.get('Train/mean dice_metric')}\n"
                )

                if config.GCNC_loader.fusion != True:
                    test_metric, _ = val_one_epoch(
                        model=model,
                        loss_params=loss_params,
                        loss_functions=loss_functions,
                        val_loader=test_loader,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        metrics=metrics,
                        post_trans=post_trans,
                        accelerator=accelerator,
                        epoch=epoch,
                        val_step=0,
                        test=True,
                    )
                    best_test_metrics = test_metric
                    test_hd95 = test_metric["Test Seg/hd95_metric"]
                else:
                    test_metric = val_metric
                    best_test_metrics = test_metric
                    test_hd95 = test_metric["Val Seg/hd95_metric"]

                # 保存模型
                accelerator.save_state(
                    output_dir=f"{os.getcwd()}/model_store/{config.finetune.GCM.checkpoint}/best"
                )

            accelerator.print(
                colored(f"Epoch [{epoch+1}", "green")
                + f"/{config.trainer.num_epochs}]  Now val Seg dice: {val_metric['Val/mean dice_metric']}, best val Seg dice: {best_val_metric_data}, best val Seg hd95: {best_val_metrics['Val Seg/hd95_metric']}, best test Seg dice: {best_test_metric_data}, best test Seg hd95: {test_hd95}\n"
            )

        # checkpoint
        accelerator.print(colored(f"Cheakpoint...", "yellow") + "\n")
        accelerator.save_state(
            output_dir=f"{os.getcwd()}/model_store/{config.finetune.GCNC.checkpoint}/checkpoint"
        )
        torch.save(
            {
                "epoch": epoch,
                "best_accuracy": best_val_metric_data,
                "best_test_accuracy": best_test_metric_data,
                "best_metrics": best_val_metrics,
                "best_test_metrics": best_test_metrics,
            },
            f"{os.getcwd()}/model_store/{config.finetune.GCNC.checkpoint}/checkpoint/epoch.pth.tar",
        )

    accelerator.print(f"best class acc: {best_test_metrics['Test/accuracy']}\n")
    accelerator.print(f"best seg acc:   {best_test_metrics['Test/dice_metric']}\n")
    accelerator.print(f"best metrics:   {best_test_metrics}")
