#!/bin/bash
python main.py  \
--input input \
--model_name resnext \
--model_depth 101 \
--resnext_cardinality 32 \
--resnet_shortcut B \
--feat_dir /home/rgh/Matches/video-caption.pytorch/data/feats/c3d_kinectics \
--video_root /home/rgh/Matches/video-caption.pytorch/data/videos \
--output output.json \
--model pretrained_models/resnext-101-64f-kinetics.pth \
--mode feature

