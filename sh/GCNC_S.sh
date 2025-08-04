export OMP_NUM_THREADS=1

export CUDA_LAUNCH_BLOCKING=1
export NCCL_DEBUG=INFO
# export CUDA_VISIBLE_DEVICES=0
export ACCELERATE_USE_DDP_FIND_UNUSED_PARAMETERS=true
torchrun \
  --nproc_per_node 4 \
  --master_port 29550 \
  GCNC_train_seg.py