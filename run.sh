export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=1,2,3
torchrun \
  --nproc_per_node 3 \
  --master_port 29551 \
  train_classify_GCM.py