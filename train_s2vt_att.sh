#!/bin/bash
LOG=log/s2vt_att-`date +%Y-%m-%d_%H-%M-%S`.log
python train.py  \
--gpu 0,1 \
--save_checkpoint_every 10 \
--epochs 1000  \
--batch_size 80  \
--input_json data/all_videodatainfo_2017.json \
--info_json data/all_info.json \
--caption_json data/all_caption.json \
--checkpoint_path checkpoint/vgg16/s2vt_att  \
--feats_dir data/feats/vgg16/trainval/  \
--dim_vid 4096 \
--rnn_type lstm \
--learning_rate_decay_every 100 \
--model S2VTAttModel \
     2>&1 | tee $LOG