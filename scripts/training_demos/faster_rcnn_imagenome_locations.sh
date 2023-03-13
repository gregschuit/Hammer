#!/bin/bash

# Help message.
if [[ $# -lt 1 ]]; then
    echo "This script launches a job of training Faster R-CNN on ImaGenome locations."
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

ANNOTATIONS_LOCATIONS_PATH_TRAIN=/home/gregschuit/projects/med-region-based-cf/data/annotations/locations_train.json
ANNOTATIONS_LOCATIONS_PATH_VAL=/home/gregschuit/projects/med-region-based-cf/data/annotations/locations_val.json
MIMIC_CXR_JPG_DIR_TRAIN=/mnt/workspace/mimic-cxr-jpg/images-256-imagenome-splits/train
MIMIC_CXR_JPG_DIR_VAL=/mnt/workspace/mimic-cxr-jpg/images-256-imagenome-splits/val

TRAIN_ANNOTATIONS=$ANNOTATIONS_LOCATIONS_PATH_TRAIN
VAL_ANNOTATIONS=$ANNOTATIONS_LOCATIONS_PATH_VAL
TRAIN_DATASET=$MIMIC_CXR_JPG_DIR_TRAIN
VAL_DATASET=$MIMIC_CXR_JPG_DIR_VAL

./scripts/dist_train.sh ${GPUS} faster_rcnn \
    --job_name='faster_rcnn_imagenome_locations' \
    --seed=0 \
    --total_epochs=20 \
    --resolution=256 \
    --image_channels=1 \
    --train_dataset=${TRAIN_DATASET} \
    --val_dataset=${VAL_DATASET} \
    --train_anno_path=${TRAIN_ANNOTATIONS} \
    --val_anno_path=${VAL_ANNOTATIONS} \
    --train_anno_format=json \
    --val_anno_format=json \
    --val_max_samples=-1 \
    --train_data_mirror=False \
    --val_data_mirror=False \
    --batch_size=6 \
    --val_batch_size=16 \
    --data_loader_type='iter' \
    --data_repeat=200 \
    --data_workers=3 \
    --data_prefetch_factor=2 \
    --data_pin_memory=true \
    --pretrained=False \
    --progress=True \
    --num_classes=36 \
    --pretrained_backbone=False \
    --trainable_backbone_layers=3 \
    --lr=0.001 \
    ${@:3}
