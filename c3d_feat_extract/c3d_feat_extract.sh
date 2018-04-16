#!/bin/bash
python main.py  \
--n_classes 20 \
--gpu 1 \
--input input \
--model_name resnext \
--model_depth 101 \
--resnext_cardinality 32 \
--resnet_shortcut B \
--feat_dir /home/rgh/Matches/video-caption.pytorch/data/feats/c3d_kinectics_msrvtt \
--video_root /home/rgh/Matches/video-caption.pytorch/data/videos \
--output output.json \
--model pretrained_models/resnext-101-MSR-VTT-finetuned-25-epochs.pth \
--mode feature


# python main.py  \
# --n_classes 101 \
# --gpu 1 \
# --input input \
# --model_name resnext \
# --model_depth 101 \
# --resnext_cardinality 32 \
# --resnet_shortcut B \
# --feat_dir /home/rgh/Matches/video-caption.pytorch/data/feats/c3d_kinectics_hmdb \
# --video_root /home/rgh/Matches/video-caption.pytorch/data/videos \
# --output output.json \
# --model pretrained_models/resnext-101-kinetics-hmdb51_split1.pth \
# --mode feature

# python main.py  \
# --n_classes 51 \
# --gpu 1 \
# --input input \
# --model_name resnext \
# --model_depth 101 \
# --resnext_cardinality 32 \
# --resnet_shortcut B \
# --feat_dir /home/rgh/Matches/video-caption.pytorch/data/feats/c3d_kinectics_ucf \
# --video_root /home/rgh/Matches/video-caption.pytorch/data/videos \
# --output output.json \
# --model pretrained_models/resnext-101-kinetics-ucf101_split1.pth \
# --mode feature

# python main.py  \
# --n_classes 400 \
# --gpu 0 \
# --input input \
# --model_name resnext \
# --model_depth 101 \
# --resnext_cardinality 32 \
# --resnet_shortcut B \
# --feat_dir /home/rgh/Matches/video-caption.pytorch/data/feats/c3d_kinectics \
# --video_root /home/rgh/Matches/video-caption.pytorch/data/videos \
# --output output.json \
# --model pretrained_models/resnext-101-kinetics.pth \
# --mode feature

# python main.py  \
# --n_classes 400 \
# --sample_duration 64 \
# --gpu 0 \
# --input input \
# --model_name resnext \
# --model_depth 101 \
# --resnext_cardinality 32 \
# --resnet_shortcut B \
# --feat_dir /home/rgh/Matches/video-caption.pytorch/data/feats/c3d_kinectics_64f \
# --video_root /home/rgh/Matches/video-caption.pytorch/data/videos \
# --output output.json \
# --model pretrained_models/resnext-101-64f-kinetics.pth \
# --mode feature

