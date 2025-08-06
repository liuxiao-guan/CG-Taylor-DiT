
  CUDA_VISIBLE_DEVICES=2 torchrun --nnodes=1 --nproc_per_node=1  --master_port=29501 sample_ddp.py \
  --model DiT-XL/2 \
  --per-proc-batch-size 50 \
  --image-size 256 \
  --cfg-scale 1.5 \
  --ddim-sample \
  --num-sampling-steps 50 \
  --interval 1 \
  --max-order 1 \
  --max_block_order 3 \
  --threshold 0.008 \
  --mid_cor 2 \


