#!/bin/bash

python caffe_feat_extract.py \
--video_path data/videos \
--output_dir data/feats/resnet269 \
--model_weight pretrained_models/resnet269-v2.caffemodel \
--model_deploy pretrained_models/deploy_resnet269-v2.prototxt \
--n_frame_steps 80  \
--gpu 1 \
--batch_size 10 \


#python caffe_feat_extract.py \
#--video_path data/videos \
#--output_dir data/feats/resnet152_places365 \
#--model_weight pretrained_models/resnet152_places365.caffemodel \
#--model_deploy pretrained_models/deploy_resnet152_places365.prototxt \
#--n_frame_steps 80  \
#--gpu 0 \
