CUDA_VISIBLE_DEVICES=3  python sample.py --ddim-sample --num-sampling-steps 50 --interval 1 --max-order 1 --max_block_order=1
CUDA_VISIBLE_DEVICES=3  python sample.py --ddim-sample --num-sampling-steps 50 --interval 1 --max-order 1 --max_block_order=1 --test-FLOPs

# CUDA_VISIBLE_DEVICES=3  python sample.py --ddim-sample --num-sampling-steps 50 --interval 1 --max-order 4
# CUDA_VISIBLE_DEVICES=3  python sample.py --ddim-sample --num-sampling-steps 50 --interval 1 --max-order 4 --test-FLOPs