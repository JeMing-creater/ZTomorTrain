export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
torchrun \
  --nproc_per_node 6 \
  --master_port 29550 \
  train_seg_GCNC.py