#!/bin/bash

python caffe_feat_extract.py \
--video_path data/videos \
--output_dir data/feats/resnet152_places365 \
--model_weight pretrained_models/resnet152_places365.caffemodel \
--model_deploy pretrained_models/deploy_resnet152_places365.prototxt \
--n_frame_steps 80  \
--gpu 0 \
