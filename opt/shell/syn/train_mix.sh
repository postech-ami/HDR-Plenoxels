#!/bin/bash
data="book_mix"
exp="book_syn_mix"
tag="book"
gpu=7

echo Launching experiment ${exp}

GPU=${gpu}
DATA_DIR=/local_data/hdr_synthetic/llff/${data}
CKPT_DIR=/local_data/ugkim/hdr_synthetic/llff/${exp}
CONFIG=configs/llff_syn.json
mkdir -p $CKPT_DIR
chmod -R 777 $CKPT_DIR
echo CKPT $CKPT_DIR
echo DATA $DATA_DIR

CUDA_VISIBLE_DEVICES=$GPU python -u hdr_opt.py \
    $DATA_DIR \
    -t $CKPT_DIR \
    -c $CONFIG \
    --exp_name ${exp} \
    --tag_name ${tag}
echo DETACH

echo TEST_VIEW_SAVE
CUDA_VISIBLE_DEVICES=$GPU python render_imgs.py \
    $CKPT_DIR \
    $DATA_DIR

echo VIDEO_SAVE
CUDA_VISIBLE_DEVICES=$GPU python render_imgs.py \
    $CKPT_DIR \
    $DATA_DIR \
    --render_path \
    --no_imsave \
    --novel_blender

echo FINISH