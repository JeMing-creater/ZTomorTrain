export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=1
# export ACCELERATE_USE_DDP_FIND_UNUSED_PARAMETERS=true
torchrun \
  --nproc_per_node 1 \
  --master_port 29550 \
  GCNC_train_seg.py