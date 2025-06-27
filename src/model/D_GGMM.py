import os
import math
from cv2 import CLAHE
import torch
import warnings
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
from mamba_ssm import Mamba
from timm.models.registry import register_model
from mmengine.model import constant_init, kaiming_init
from timm.models.layers import DropPath, to_2tuple, make_divisible, trunc_normal_

warnings.filterwarnings("ignore")


class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        ctx.save_for_backward(i)
        return i * torch.sigmoid(i)

    @staticmethod
    def backward(ctx, grad_output):
        sigmoid_i = torch.sigmoid(ctx.saved_variables[0])
        return grad_output * (
            sigmoid_i * (1 + ctx.saved_variables[0] * (1 - sigmoid_i))
        )


# Swish激活函数
class Swish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)


class Mlp(nn.Module):
    def __init__(self, dim, shallow=False):
        super().__init__()
        drop = 0.0
        self.fc1 = nn.Conv3d(dim, dim * 4, 1)
        self.dwconv = nn.Conv3d(dim * 4, dim * 4, 3, 1, 1, bias=True, groups=dim * 4)
        if shallow == True:
            self.act = nn.GELU()
        else:
            self.act = Swish()
        self.fc2 = nn.Conv3d(dim * 4, dim, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DLK(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.att_conv1 = nn.Conv3d(
            dim, dim, kernel_size=5, stride=1, padding=2, groups=dim
        )
        self.att_conv2 = nn.Conv3d(
            dim, dim, kernel_size=7, stride=1, padding=9, groups=dim, dilation=3
        )

        self.spatial_se = nn.Sequential(
            nn.Conv3d(in_channels=2, out_channels=2, kernel_size=7, padding=3),
            nn.Sigmoid(),
        )

    def forward(self, x):
        att1 = self.att_conv1(x)
        att2 = self.att_conv2(att1)

        att = torch.cat([att1, att2], dim=1)
        avg_att = torch.mean(att, dim=1, keepdim=True)
        max_att, _ = torch.max(att, dim=1, keepdim=True)
        att = torch.cat([avg_att, max_att], dim=1)
        att = self.spatial_se(att)
        output = att1 * att[:, 0, :, :].unsqueeze(1) + att2 * att[:, 1, :, :].unsqueeze(
            1
        )
        output = output + x
        return output


class DLKModule(nn.Module):
    def __init__(self, dim, num_slices=4, shallow=True):
        super().__init__()

        self.proj_1 = nn.Conv3d(dim, dim, 1)
        # self.act = nn.GELU()
        # self.spatial_gating_unit = DLK(dim)

        self.spatial_gating_unit = GGM_Block(
            in_dim=dim,
            out_dim=dim,
            shallow=shallow,
            num_slices=num_slices,
        )

        self.proj_2 = nn.Conv3d(dim, dim, 1)

    def forward(self, x):
        shortcut = x.clone()
        x = self.proj_1(x)
        # x = self.act(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shortcut
        return x


class GGM_Module(nn.Module):
    def __init__(self, dim, out_dim=0, num_slices=4, shallow=True):
        super().__init__()

        self.proj_1 = nn.Conv3d(dim, dim, 1)
        self.act = nn.GELU()
        # self.spatial_gating_unit = DLK(dim)

        if out_dim != 0:
            self.spatial_gating_unit = GGM_Block(
                in_dim=dim,
                out_dim=out_dim,
                shallow=shallow,
                num_slices=num_slices,
            )
            dim = out_dim
        else:
            self.spatial_gating_unit = GGM_Block(
                in_dim=dim,
                out_dim=dim,
                shallow=shallow,
                num_slices=num_slices,
            )
            dim = dim

        self.proj_2 = nn.Conv3d(dim, dim, 1)

        self.mlp = Mlp(dim, shallow)
        self.out_dim = out_dim

    def forward(self, x):
        shortcut = x.clone()
        x = self.proj_1(x)
        x = self.act(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        if self.out_dim == 0:
            x = x + shortcut
        x = self.mlp(x)
        return x


class DLKBlock(nn.Module):
    def __init__(self, dim, num_slices=4, shallow=False, drop_path=0.0):
        super().__init__()
        self.norm_layer = nn.LayerNorm(dim, eps=1e-6)
        self.attn = DLKModule(dim, num_slices=num_slices, shallow=shallow)
        self.mlp = Mlp(dim, shallow)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        layer_scale_init_value = 1e-6
        self.layer_scale = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True
        )

    def forward(self, x):
        shortcut = x.clone()
        x = channel_to_last(x)
        x = self.norm_layer(x)
        x = channel_to_first(x)
        x = self.attn(x)
        x = shortcut + self.drop_path(
            self.layer_scale.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * x
        )

        shortcut = x.clone()
        x = channel_to_last(x)
        x = self.norm_layer(x)
        x = channel_to_first(x)
        x = self.mlp(x)
        x = shortcut + self.drop_path(
            self.layer_scale.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * x
        )

        return x


class Encoder(nn.Module):
    def __init__(self, in_chans, num_slices_list, depths, dims, drop_path_rate):
        super().__init__()

        self.downsample_layers = nn.ModuleList()
        stem = nn.Conv3d(in_chans, dims[0], kernel_size=7, stride=2, padding=3)
        self.downsample_layers.append(stem)

        for i in range(3):
            downsample_layer = nn.Conv3d(dims[i], dims[i + 1], kernel_size=2, stride=2)
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[
                    # DLKBlock(
                    #     dim=dims[i],
                    #     num_slices=num_slices_list[i],
                    #     drop_path=dp_rates[cur + j],
                    #     shallow=shallow,
                    # )
                    GGM_Module(dim=dims[i], 
                               num_slices=num_slices_list[i], 
                               shallow=(True if i<=1 else False))
                    for j in range(depths[i])
                ]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm_layers = nn.ModuleList()
        for i in range(4):
            norm_layer = nn.LayerNorm(dims[i], eps=1e-6)
            self.norm_layers.append(norm_layer)

    def forward_features(self, x):
        outs = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = channel_to_last(x)
            x = self.norm_layers[i](x)
            x = channel_to_first(x)
            x = self.stages[i](x)
            outs.append(x)

        return tuple(outs)

    def forward(self, x):
        x = self.forward_features(x)
        return x


class Convblock(nn.Module):
    def __init__(self, input_dim, dim, shallow=False):
        super().__init__()
        if shallow == True:
            self.act = nn.GELU()
        else:
            self.act = Swish()
        self.conv1 = nn.Sequential(
            nn.Conv3d(input_dim, dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(dim),
            self.act,
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(dim),
            self.act,
        )

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2(output)
        return output


def channel_to_last(x):
    """
    Args:
        x: (B, C, H, W, D)

    Returns:
        x: (B, H, W, D, C)
    """
    return x.permute(0, 2, 3, 4, 1)


def channel_to_first(x):
    """
    Args:
        x: (B, H, W, D, C)

    Returns:
        x: (B, C, H, W, D)
    """
    return x.permute(0, 4, 1, 2, 3)


class GGM(nn.Module):
    def __init__(
        self,
        in_dim=1,
        out_dim=32,
        d_state=16,
        d_conv=4,
        expand=2,
        head=4,
        num_slices=4,
        step=1,
        goble=True,
    ):
        super(GGM, self).__init__()

        # 大核全局扫描
        if goble == True:
            # 大核全局扫描
            self.att_conv = nn.Conv3d(
                in_dim,
                in_dim,
                kernel_size=7,
                stride=1,
                padding=9,
                groups=in_dim,
                dilation=3,
            )
        else:
            # 小核局部扫描
            self.att_conv = nn.Conv3d(
                in_dim, in_dim, kernel_size=5, stride=1, padding=2, groups=in_dim
            )

        self.spatial_se = nn.Sequential(
            nn.Conv3d(in_channels=2, out_channels=2, kernel_size=7, padding=3),
            nn.Sigmoid(),
        )

        # Mamba
        self.norm = nn.LayerNorm(in_dim)
        self.mamba = Mamba(
            d_model=in_dim,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor
            bimamba_type="v2",  # TODO: set 154 assert bimamba_type=="v3" as none
            nslices=num_slices,
        )
        # none requires_grad
        self.mamba.A_s_log = None
        self.mamba.D_s = None
        self.mamba.conv1d_s = None
        self.mamba.x_proj_s = None
        self.mamba.dt_proj_s = None

        # 调整通道Conv
        # self.conv = nn.Conv3d(in_dim, out_dim, 3, 1, 1)

    def forward(self, x):

        # 保持原始x
        att1 = x

        # 全局特征提取
        x = self.att_conv(x)

        # 维度记录
        B, C, H, W, D = x.shape

        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]

        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)

        # x_mamba, o_1, o_2, o_3 = self.mamba(x_norm)
        out, q, k, v = self.mamba(x_norm)

        # 维度转换
        att2 = out.transpose(-1, -2).reshape(B, C, *img_dims)

        # 调整通道
        att = torch.cat([att1, att2], dim=1)
        avg_att = torch.mean(att, dim=1, keepdim=True)
        max_att, _ = torch.max(att, dim=1, keepdim=True)
        att = torch.cat([avg_att, max_att], dim=1)
        att = self.spatial_se(att)
        output = att1 * att[:, 0, :, :, :].unsqueeze(1) + att2 * att[
            :, 1, :, :, :
        ].unsqueeze(1)

        # x = self.conv(output)
        x = output

        return x, q, k, v


class GGM_Block(nn.Module):
    def __init__(self, in_dim=3, out_dim=32, num_slices=4, shallow=True):
        super(GGM_Block, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        if shallow == True:
            self.act = nn.GELU()
        else:
            self.act = Swish()

        self.gobel_attention = GGM(
            in_dim=in_dim // 2, out_dim=out_dim, num_slices=num_slices, goble=True
        )

        self.local_attention = GGM(
            in_dim=in_dim // 2, out_dim=out_dim, num_slices=num_slices, goble=False
        )

        self.downsample = Convblock(in_dim * 2, out_dim, shallow=shallow)
        self.shallow = shallow
        if self.shallow == True:
            self.pool = nn.MaxPool1d(kernel_size=8, stride=8)
            self.up = nn.Conv3d(
                in_channels=out_dim // 8, out_channels=out_dim, kernel_size=1
            )

    def auto_shape(self, N):
        cube_root = round(N ** (1 / 3))
        for d in range(cube_root, 0, -1):
            if N % d != 0:
                continue
            rest = N // d
            for h in range(int(math.sqrt(rest)), 0, -1):
                if rest % h == 0:
                    w = rest // h
                    return (d, h, w)

    def forward(self, x):
        x_0, x_1 = x.chunk(2, dim=1)
        x_0, _, k, v = self.gobel_attention(x_0)
        x_1, _, q, _ = self.local_attention(x_1)

        B, C, H, W, D = x.shape
        # q, k, v = q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1)
        if self.shallow == True:
            b_s, c_s, n = q.shape  # 记录原始尺寸，让out_a转换成原始尺寸
            q = self.pool(q)
            k = self.pool(k)
            v = self.pool(v)

        q, k, v = q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1)
        attn = (q.transpose(-2, -1) @ k).softmax(-1)
        out_a = v @ attn.transpose(-2, -1)

        if self.shallow == True:
            d, h, w = self.auto_shape(n)
            # out_a = F.interpolate(out_a, size=n, mode="nearest")
            out_a = out_a.view(B, -1, h, w, d)
            out_a = self.up(out_a)
        else:
            out_a = out_a.view(B, -1, H, W, D)

        x_f = torch.cat([x_0, x_1], dim=1)

        x = torch.cat([x_f, out_a], dim=1)

        x = self.downsample(x)
        return x


class Seg_Decoder(nn.Module):
    def __init__(
        self,
        out_channels=1,
        dims=[48, 96, 192, 384],
        out_dim=64,
        num_slices_list=[64, 32, 16, 8],
    ):
        super().__init__()
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = (
            dims[0],
            dims[1],
            dims[2],
            dims[3],
        )

        # self.block2 = GGM_Block(
        #     in_dim=c2_in_channels,
        #     out_dim=out_dim,
        #     shallow=False,
        #     # shallow=True,
        #     num_slices=num_slices_list[1],
        # )
        self.block2 = GGM_Module(
            dim=c2_in_channels,
            out_dim=out_dim,
            num_slices=num_slices_list[1],
            shallow=False,
        )
        # self.block3 = GGM_Block(
        #     in_dim=c3_in_channels,
        #     out_dim=out_dim,
        #     shallow=False,
        #     num_slices=num_slices_list[2],
        # )
        self.block3 = GGM_Module(
            dim=c3_in_channels,
            out_dim=out_dim,
            num_slices=num_slices_list[2],
            shallow=False,
        )
        # self.block4 = GGM_Block(
        #     in_dim=c4_in_channels,
        #     out_dim=out_dim,
        #     shallow=False,
        #     num_slices=num_slices_list[3],
        # )
        self.block4 = GGM_Module(
            dim=c4_in_channels,
            out_dim=out_dim,
            num_slices=num_slices_list[3],
            shallow=False,
        )

        self.fuse = Convblock(out_dim, out_dim, shallow=True)
        self.fuse2 = nn.Sequential(
            Convblock(out_dim * 2, out_dim, shallow=False),
            nn.Conv3d(out_dim, out_dim, kernel_size=1, bias=False),
        )

        self.L_feature = Convblock(c1_in_channels, out_dim, shallow=True)

        self.o1_u = nn.ConvTranspose3d(out_dim, out_dim, kernel_size=4, stride=4)
        self.o2_u = nn.ConvTranspose3d(out_dim * 2, out_dim, kernel_size=2, stride=2)

        self.SEG_head = nn.Conv3d(out_dim * 2, out_channels, kernel_size=1, bias=False)

    def Upsample(self, x, size, align_corners=False):
        """
        Wrapper Around the Upsample Call
        """
        return nn.functional.interpolate(
            x, size=size, mode="trilinear", align_corners=align_corners
        )

    def forward(self, encode_dim):
        c1, c2, c3, c4 = encode_dim
        _c4 = self.block4(c4)
        _c4 = self.Upsample(_c4, c3.size()[2:])
        _c3 = self.block3(c3)
        _c2 = self.block2(c2)

        output = self.fuse2(
            torch.cat(
                [self.Upsample(_c4, c2.size()[2:]), self.Upsample(_c3, c2.size()[2:])],
                dim=1,
            )
        )

        L_feature = self.L_feature(c1)  # [1, 64, 88, 88]
        H_feature = self.fuse(_c2)
        H_feature = self.Upsample(H_feature, L_feature.size()[2:])

        output2 = torch.cat((H_feature, L_feature), dim=1)

        output = self.o1_u(output)
        output2 = self.o2_u(output2)

        SEG_out = self.SEG_head(torch.cat((output, output2), dim=1))

        return SEG_out
        return x


class Class_Decoder(nn.Module):

    def __init__(self, dim=384, class1_channels=1, class2_channels=None):
        super().__init__()
        # 添加全局平均池化层
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        # # 最后一层全连接层用于分类任务
        # # 为每个任务定义独立的分类头
        # self.task_heads = nn.ModuleList([
        #     nn.Sequential(
        #         nn.Linear(48*2, 64),  # 全连接层
        #         nn.ReLU(),
        #         nn.Linear(64, 1)  # 输出层
        #     ) for _ in range(num_tasks)
        # ])
        self.task1_head = nn.Sequential(
            nn.Linear(dim, 64),
            nn.ReLU(),
            nn.Linear(64, class1_channels),  # 全连接层  # 输出层
        )
        self.mutil_task = False
        if class2_channels != None:
            self.task2_head = nn.Sequential(
                nn.Linear(dim, 64),
                nn.ReLU(),
                nn.Linear(64, class2_channels),  # 全连接层  # 输出层
            )
            self.mutil_task = True

    def forward(self, hidden_downsample):

        # 展平特征并送入全连接层
        features = self.global_avg_pool(hidden_downsample)
        features = torch.flatten(features, start_dim=1)
        # 对每个任务应用独立的分类头，并收集结果
        # task_outputs = [task_head(features) for task_head in self.task_heads]
        task1_outputs = self.task1_head(features)
        if self.mutil_task == True:
            task2_outputs = self.task2_head(features)
            # 在channels维度上拼接所有任务的输出

            return task1_outputs, task2_outputs
        else:
            return task1_outputs, None


class D_GGMM(nn.Module):

    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        class1_channels=1,
        class2_channels=1,
        depths=[2, 2, 2, 2],
        dims=[48, 96, 192, 384],
        kernel_size=3,
        out_dim=64,
        num_slices_list=[64, 32, 16, 8],
        drop_path_rate=0.3,
    ):
        super().__init__()

        self.dnet_down = Encoder(
            in_chans=in_channels,
            depths=depths,
            dims=dims,
            num_slices_list=num_slices_list,
            drop_path_rate=drop_path_rate,
        )

        self.seg_decoder = Seg_Decoder(
            out_channels=out_channels,
            dims=dims,
            out_dim=out_dim,
            num_slices_list=num_slices_list,
        )

        self.class_decoder = Class_Decoder(
            dim=dims[-1],
            class1_channels=class1_channels,
            class2_channels=class2_channels,
        )

    def Upsample(self, x, size, align_corners=False):
        """
        Wrapper Around the Upsample Call
        """
        return nn.functional.interpolate(
            x, size=size, mode="trilinear", align_corners=align_corners
        )

    def forward(self, x):
        c1, c2, c3, c4 = self.dnet_down(x)

        CLS_out = self.class_decoder(c4)

        SEG_out = self.seg_decoder((c1, c2, c3, c4))

        return CLS_out, SEG_out


def benchmark_model(
    model: nn.Module, input_size=(1, 3, 224, 224), device="cuda", warmup=10, reps=100
):
    import time
    from thop import profile, clever_format
    from ptflops import get_model_complexity_info

    """
    计算 PyTorch 模型的 FPS（Frames Per Second）、参数量（MParam）和计算量（GLOPs）
    :param model: PyTorch nn.Module
    :param input_size: 输入张量的尺寸 (B, C, H, W)
    :param device: 'cuda' 或 'cpu'
    :param warmup: 预热轮数
    :param reps: 正式测试轮数
    :return: dict 包含 FPS、MParams、GLOPs
    """
    model = model.to(device)
    model.eval()

    dummy_input = torch.randn(*input_size).to(device)

    # ----------------------- 计算 FLOPs 和 参数量 -----------------------
    with torch.no_grad():
        macs, params = profile(model, inputs=(dummy_input,), verbose=False)
        macs, params = clever_format([macs, params], "%.3f")

    # ----------------------- 计算 FPS -----------------------
    with torch.no_grad():
        # 预热
        for _ in range(warmup):
            _ = model(dummy_input)
        torch.cuda.synchronize()

        start = time.time()
        for _ in range(reps):
            _ = model(dummy_input)
        torch.cuda.synchronize()
        end = time.time()

        fps = reps / (end - start)
    print({"FPS": round(fps, 2), "Params": params, "GLOPs": macs})
    # return {"FPS": round(fps, 2), "Params": params, "GLOPs": macs}


if __name__ == "__main__":
    device = "cuda:0"

    x = torch.randn(size=(2, 3, 128, 128, 64)).to(device)
    # test_x = torch.randn(size=(2, 64, 88, 88)).to(device)

    model = D_GGMM(
        in_channels=3, out_channels=3, class1_channels=1, class2_channels=1
    ).to(device)

    CLS_out, SEG_out = model(x)

    print(CLS_out[0].size())
    print(CLS_out[1].size())
    print(CLS_out[0])
    print(CLS_out[1])

    print(SEG_out.size())

    # benchmark_model(
    #     model,
    #     input_size=(1, 3, 608, 608),
    #     device=device,
    #     warmup=10,
    #     reps=100,
    # )
    # print(module(test_x).size())
