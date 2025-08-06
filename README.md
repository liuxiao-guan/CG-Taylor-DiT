
This is the implementation of CG-Taylor for  Accelerating  Diffusion Models.

[Forecasting When to Forecast: Accelerating Diffusion Models with Confidence-Gated Taylor](http://arxiv.org/abs/2508.02240)

If you have any question, please contact [liuxiaoguan@whu.edu.cn](liuxiaoguan@whu.edu.cn).
### 1. Prepare Environment

```bash
cd TaylorSeer-DiT
conda env create -f environment.yml
conda activate DiT
pip install flash-attention
```

### 2. Download Checkpoints

Simply follow the official documentation to download the necessary checkpoints.

### 3. Run Samples

#### Single-Batch Inference
Set your desired class ID in `sample.py`.

#### Distributed Data Parallel (DDP) Inference
```bash
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

```
### 4. Acknowlege
The code is based on [TaylorSeer-DiT](https://github.com/Shenyi-Z/TaylorSeer/tree/main/TaylorSeer-DiT).
