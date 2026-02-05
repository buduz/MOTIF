#!/bin/bash

PROJECT_ROOT="/home/zhiheng/project/MOTIF"
OUTPUT_BASE="$PROJECT_ROOT/outputs"
mkdir -p "$OUTPUT_BASE"

device=cuda:3
stage1_dir="$OUTPUT_BASE/outputs_stage1"
stage2_dir="$OUTPUT_BASE/outputs_stage2"
stage3_dir="$OUTPUT_BASE/outputs_stage3"

# train stage 1
python "$PROJECT_ROOT/src/train_stage_1.py" \
    device=$device \
    output_dir=$stage1_dir \
    dataset.num_workers=16 \
    train.epochs=20 \
    batch_size=128 \
    num_robots=2
sleep 10

# train stage 2
python "$PROJECT_ROOT/src/train_stage_2.py" \
    device=$device \
    output_dir="$stage2_dir" \
    models.stage2.vq_vae_ckpt="$stage1_dir/epoch_0020.pth" \
    train.epochs=30 \
    batch_size=128
sleep 10

# train stage 3
python "$PROJECT_ROOT/src/train_stage_3.py" \
    device=$device \
    output_dir="$stage3_dir" \
    models.stage2.vq_vae_ckpt="$stage1_dir/epoch_0020.pth" \
    models.stage3.mkl_ckpt="$stage2_dir/epoch_0030.pth" \
    train.epochs=60 \
    batch_size=64