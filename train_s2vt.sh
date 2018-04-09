#!/bin/bash

### nasnet_resnet101
feat=nasnet_resnet101
LOG=log/s2vt_${feat}_40frames-`date +%Y-%m-%d_%H-%M-%S`.log
python train.py  \
--gpu 0 \
--save_checkpoint_every 20 \
--epochs 1000  \
--n_frame_steps 40 \
--batch_size 100  \
--input_json data/all_videodatainfo_2017.json \
--info_json data/all_info.json \
--caption_json data/all_caption.json \
--checkpoint_path checkpoint/${feat}_40frames/s2vt  \
--feats_dir data/feats/nasnet  data/feats/resnet101 \
--dim_vid 6080 \
--rnn_type lstm \
--learning_rate_decay_every 100 \
--model S2VTModel \
     2>&1 | tee $LOG

# ### nasnet
# feat=nasnet
# LOG=log/s2vt_${feat}_40frames-`date +%Y-%m-%d_%H-%M-%S`.log
# python train.py  \
# --gpu 0 \
# --save_checkpoint_every 50 \
# --epochs 1000  \
# --n_frame_steps 40 \
# --batch_size 100  \
# --input_json data/all_videodatainfo_2017.json \
# --info_json data/all_info.json \
# --caption_json data/all_caption.json \
# --checkpoint_path checkpoint/${feat}_40frames/s2vt  \
# --feats_dir data/feats/$feat/  \
# --dim_vid 4032 \
# --rnn_type lstm \
# --learning_rate_decay_every 100 \
# --model S2VTModel \
#      2>&1 | tee $LOG


###  inception_v4 40frames
# feat=inception_v4
# LOG=log/s2vt_${feat}_40frames-`date +%Y-%m-%d_%H-%M-%S`.log
# python train.py  \
# --gpu 0 \
# --save_checkpoint_every 50 \
# --epochs 1000  \
# --n_frame_steps 40 \
# --batch_size 100  \
# --input_json data/all_videodatainfo_2017.json \
# --info_json data/all_info.json \
# --caption_json data/all_caption.json \
# --checkpoint_path checkpoint/${feat}_40frames/s2vt  \
# --feats_dir data/feats/$feat/  \
# --dim_vid 1536 \
# --rnn_type lstm \
# --learning_rate_decay_every 100 \
# --model S2VTModel \
#      2>&1 | tee $LOG

###  resnet101 40frames
# feat=resnet101
# LOG=log/s2vt_${feat}_40frames-`date +%Y-%m-%d_%H-%M-%S`.log
# python train.py  \
# --gpu 1 \
# --save_checkpoint_every 50 \
# --epochs 1000  \
# --n_frame_steps 40 \
# --batch_size 100  \
# --input_json data/all_videodatainfo_2017.json \
# --info_json data/all_info.json \
# --caption_json data/all_caption.json \
# --checkpoint_path checkpoint/${feat}_40frames/s2vt  \
# --feats_dir data/feats/$feat/  \
# --dim_vid 2048 \
# --rnn_type lstm \
# --learning_rate_decay_every 100 \
# --model S2VTModel \
#      2>&1 | tee $LOG


## resnet101_c3d_fc7_wo_ft
# feat=resnet101
# LOG=log/s2vt_resnet101_c3d_fc7_wo_ft-`date +%Y-%m-%d_%H-%M-%S`.log
# python train.py  \
# --gpu 1 \
# --save_checkpoint_every 50 \
# --epochs 1000  \
# --n_frame_steps 80 \
# --batch_size 100  \
# --input_json data/all_videodatainfo_2017.json \
# --info_json data/all_info.json \
# --caption_json data/all_caption.json \
# --checkpoint_path checkpoint/resnet101_c3d_fc7_wo_ft/s2vt  \
# --feats_dir data/feats/$feat/  \
# --dim_vid 6144 \
# --with_c3d 1 \
# --c3d_feats_dir data/feats/c3d_fc7_wo_ft \
# --rnn_type lstm \
# --learning_rate_decay_every 200 \
# --model S2VTModel \
#      2>&1 | tee $LOG

# ###  resnet101 80frames
# feat=resnet101
# LOG=log/s2vt_${feat}_80frames-`date +%Y-%m-%d_%H-%M-%S`.log
# python train.py  \
# --gpu 0 \
# --save_checkpoint_every 50 \
# --epochs 500  \
# --n_frame_steps 80 \
# --batch_size 200  \
# --input_json data/all_videodatainfo_2017.json \
# --info_json data/all_info.json \
# --caption_json data/all_caption.json \
# --checkpoint_path checkpoint/$feat/s2vt  \
# --feats_dir data/feats/$feat/  \
# --dim_vid 2048 \
# --rnn_type lstm \
# --learning_rate_decay_every 100 \
# --model S2VTModel \
#      2>&1 | tee $LOG