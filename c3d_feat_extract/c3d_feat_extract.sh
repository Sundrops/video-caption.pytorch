#!/bin/bash
python main.py  \
--gpu 1 \
--input input \
--model_name resnext \
--model_depth 101 \
--resnext_cardinality 32 \
--resnet_shortcut B \
--feat_dir /home/rgh/Matches/video-caption.pytorch/data/feats/c3d_kinectics_64f \
--video_root /home/rgh/Matches/video-caption.pytorch/data/videos \
--output output.json \
--model pretrained_models/resnext-101-64f-kinetics.pth \
--mode feature


#python main.py  \
#--gpu 0 \
#--input input \
#--model_name resnext \
#--model_depth 101 \
#--resnext_cardinality 32 \
#--resnet_shortcut B \
#--feat_dir /home/rgh/Matches/video-caption.pytorch/data/feats/c3d_kinectics_hmdb_64f \
#--video_root /home/rgh/Matches/video-caption.pytorch/data/videos \
#--output output.json \
#--model pretrained_models/resnext-101-64f-kinetics-hmdb51_split1.pth \
#--mode feature
