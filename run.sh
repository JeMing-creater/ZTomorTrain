export OMP_NUM_THREADS=1
# export CUDA_VISIBLE_DEVICES=0,3,5,7
torchrun \
  --nproc_per_node 8 \
  --master_port 29550 \
  main_class.py