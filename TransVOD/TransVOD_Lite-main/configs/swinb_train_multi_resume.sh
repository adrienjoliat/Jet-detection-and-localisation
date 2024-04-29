#!/usr/bin/env bash

set -x
T=`date +%m%d%H%M`

EXP_DIR=exps/multi_model_jet/essai2_2
mkdir -p ${EXP_DIR}
PY_ARGS=${@:1}
python -u main.py \
    --backbone swin_b_p4w7 \
    --epochs 7 \
    --num_feature_levels 1 \
    --num_queries 100 \
    --dilation \
    --batch_size 1 \
    --num_frames 12 \
    --hidden_dim 256 \
    --lr_drop_epochs 5 6 \
    --num_workers 1 \
    --with_box_refine \
    --dataset_file 'vid_multi' \
    --resume exps/multi_model_jet/essai2/checkpoint0000.pth \
    --start_epoch 1 \
    --output_dir ${EXP_DIR} \
    ${PY_ARGS} 2>&1 | tee ${EXP_DIR}/log.train.$T
