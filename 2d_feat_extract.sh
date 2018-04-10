#!/bin/bash
feat=nasnet
python prepro_feats.py \
--video_path data/videos \
--model ${feat} \
--output_dir data/feats/${feat} \
--n_frame_steps 80  \
--gpu 0 \


# --saved_model pretrain_models/resnet152-b121ed2d.pth \
# vgg16-397923af.pth
# resnet101-5d3b4d8f.pth
# resnet152-b121ed2d.pth