export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0,6
torchrun \
  --nproc_per_node 2 \
  --master_port 29550 \
  main_class_seg.py