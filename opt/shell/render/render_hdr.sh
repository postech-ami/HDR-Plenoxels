#!/bin/bash
gpu=7
data="book_mix"
exp="book_syn_tone"
root="hdr_synthetic"

CUDA_VISIBLE_DEVICES=${gpu} python hdr_render_imgs.py \
    /local_data/ugkim/${root}/llff/${exp} \
    /local_data/${root}/llff/${data} \
    --tone_mapping "piece_wise"
