#!/bin/bash

# Help message.
if [[ $# -lt 1 ]]; then
    echo "This script launches a job of training StyleGAN2 on MIMIC-CXR."
    echo
    echo "Note: All settings are already preset for training with 8 GPUs." \
         "Please pass addition options, which will overwrite the original" \
         "settings, if needed."
    echo
    echo "Usage: $0 GPUS [OPTIONS]"
    echo
    echo "Example: $0 8 [--help]"
    echo
    exit 0
fi

GPUS=$1
TRAIN_DATASET=/home/gregschuit/data/challenge_data
VAL_DATASET=/home/gregschuit/data/challenge_data

./scripts/dist_train.sh ${GPUS} stylegan2 \
    --job_name='stylegan2_dgm_challenge_cont_2e-4' \
    --seed=0 \
    --resolution=512 \
    --image_channels=1 \
    --train_dataset=${TRAIN_DATASET} \
    --val_dataset=${VAL_DATASET} \
    --val_max_samples=-1 \
    --total_epochs=2 \
    --batch_size=4 \
    --val_batch_size=16 \
    --train_data_mirror=true \
    --data_loader_type='iter' \
    --data_repeat=200 \
    --data_workers=3 \
    --data_prefetch_factor=2 \
    --data_pin_memory=true \
    --train_data_file_format='png_dir' \
    --val_data_file_format='png_dir' \
    --g_init_res=4 \
    --latent_dim=512 \
    --d_fmaps_factor=1.0 \
    --g_fmaps_factor=1.0 \
    --d_mbstd_groups=4 \
    --g_num_mappings=8 \
    --d_lr=0.0002 \
    --g_lr=0.0002 \
    --w_moving_decay=0.995 \
    --sync_w_avg=false \
    --style_mixing_prob=0.9 \
    --r1_interval=16 \
    --r1_gamma=10.0 \
    --pl_interval=4 \
    --pl_weight=2.0 \
    --pl_decay=0.01 \
    --pl_batch_shrink=2 \
    --g_ema_img=10_000 \
    --g_ema_rampup=0.0 \
    --eval_at_start=true \
    --eval_interval=6400 \
    --ckpt_interval=6400 \
    --log_interval=128 \
    --enable_amp=false \
    --use_ada=false \
    --num_fp16_res=0 \
    --weight_path=/home/gregschuit/projects/Hammer/repo/work_dirs/stylegan2_dgm_challenge/checkpoints/checkpoint-054265.pth \
    ${@:4}
