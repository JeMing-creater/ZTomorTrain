export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0,1,2
# export ACCELERATE_USE_DDP_FIND_UNUSED_PARAMETERS=true
torchrun \
  --nproc_per_node 3 \
  --master_port 29552 \
  GCNC_train_classify.py