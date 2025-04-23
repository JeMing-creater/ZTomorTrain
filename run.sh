export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7
torchrun \
  --nproc_per_node 6 \
  --master_port 29550 \
  train_class.py