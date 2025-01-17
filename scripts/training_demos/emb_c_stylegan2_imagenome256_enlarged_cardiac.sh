#!/bin/bash

# Help message.
if [[ $# -lt 1 ]]; then
    echo "This script launches a job for training Region-guided Conditional StyleGAN2 on MIMIC-CXR."
    echo
    echo "    - Region model: YOLOv8 (ultralytics)"
    echo "    - Resolution: 256"
    echo "    - Conditioning on: enlarged cardiac silhouette (~16% of total images), single binary condition"
    echo "    - Views: PA"
    echo "    - Squaring: resized"
    echo "    - Balance: slight imbalance (enlarged cardiac silhouette ~16%)."
    echo
    echo "Usage: $0 GPUS [OPTIONS]"
    echo
    echo "Example: $0 8 [--help]"
    echo
    exit 0
fi

GPUS=$1

MIMIC_CXR_JPG_DIR=/mnt/workspace/mimic-cxr-jpg/images-small
TRAIN_ANNOTATIONS=/home/gregschuit/projects/cxr-data/annotations/enlarged_cardiac_PA_train.txt
VALID_ANNOTATIONS=/home/gregschuit/projects/cxr-data/annotations/enlarged_cardiac_PA_valid.txt

# TRAIN_ANNOTATIONS=/home/gregschuit/projects/cxr-data/annotations/dev64.txt
# VALID_ANNOTATIONS=/home/gregschuit/projects/cxr-data/annotations/dev64.txt

CLASSIFIER_WEIGTHS_PATH=/home/gregschuit/projects/thesis/results/cxr_classification/enlarged_cardiac_PA/densenet121_0/checkpoints/best-epoch-3.pth

./scripts/dist_train.sh ${GPUS} embed_stylegan2 \
    --job_name='emb_c_stylegan2_imagenome256_enlarged_cardiac_pa_resized_refactored_continue_4' \
    --classifier_weights_path=${CLASSIFIER_WEIGTHS_PATH} \
    --use_region_based_loss=false \
    --seed=0 \
    --resolution=256 \
    --image_channels=1 \
    --train_dataset=${MIMIC_CXR_JPG_DIR} \
    --val_dataset=${MIMIC_CXR_JPG_DIR} \
    --train_anno_path=${TRAIN_ANNOTATIONS} \
    --val_anno_path=${VALID_ANNOTATIONS} \
    --train_anno_format=txt \
    --val_anno_format=txt \
    --val_max_samples=-1 \
    --total_img=300_000 \
    --batch_size=4 \
    --val_batch_size=16 \
    --train_data_mirror=false \
    --data_loader_type='iter' \
    --data_repeat=200 \
    --data_workers=20 \
    --data_prefetch_factor=2 \
    --data_pin_memory=true \
    --train_data_file_format='jpg_dir' \
    --val_data_file_format='jpg_dir' \
    --g_init_res=4 \
    --latent_dim=512 \
    --d_fmaps_factor=1.0 \
    --g_fmaps_factor=1.0 \
    --d_mbstd_groups=4 \
    --g_num_mappings=8 \
    --d_lr=0.002 \
    --g_lr=0.002 \
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
    --label_dim=2 \
    --resume_path=/home/gregschuit/projects/Hammer/work_dirs/emb_c_stylegan2_imagenome256_enlarged_cardiac_pa_resized_refactored_continue_3/checkpoints/checkpoint-032000.pth \
    ${@:4}
