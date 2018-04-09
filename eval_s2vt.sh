#!/bin/bash

###  nasnet_resnet101_40frames
feat=nasnet_resnet101_40frames
epoch=60
python eval.py  \
--rnn_type lstm \
--results_path result/$feat/s2vt \
--recover_opt checkpoint/$feat/s2vt/opt_info.json \
--saved_model checkpoint/$feat/s2vt/model_$epoch.pth \
--batch_size 100 \
--gpu 0



# ###  nasnet_40frames
# feat=nasnet_40frames
# epoch=250
# python eval.py  \
# --rnn_type lstm \
# --results_path result/$feat/s2vt \
# --recover_opt checkpoint/$feat/s2vt/opt_info.json \
# --saved_model checkpoint/$feat/s2vt/model_$epoch.pth \
# --batch_size 100 \
# --gpu 0


###  inception_v4 40frames
# feat=inception_v4_40frames
# epoch=300
# python eval.py  \
# --rnn_type lstm \
# --results_path result/$feat/s2vt \
# --recover_opt checkpoint/$feat/s2vt/opt_info.json \
# --saved_model checkpoint/$feat/s2vt/model_$epoch.pth \
# --batch_size 100 \
# --gpu 0


# feat=resnet101_40frames
# epoch=150
# python eval.py  \
# --rnn_type lstm \
# --results_path result/$feat/s2vt \
# --recover_opt checkpoint/$feat/s2vt/opt_info.json \
# --saved_model checkpoint/$feat/s2vt/model_$epoch.pth \
# --batch_size 100 \
# --gpu 1


# feat=resnet101_c3d_fc7_wo_ft
# epoch=150
# python eval.py  \
# --rnn_type lstm \
# --results_path result/$feat/s2vt \
# --recover_opt checkpoint/$feat/s2vt/opt_info.json \
# --saved_model checkpoint/$feat/s2vt/model_$epoch.pth \
# --batch_size 100 \
# --gpu 1


# feat=resnet101_80frames
# feat=resnet101
# epoch=150
# python eval.py  \
# --rnn_type lstm \
# --results_path result/$feat/s2vt \
# --recover_opt checkpoint/$feat/s2vt/opt_info.json \
# --saved_model checkpoint/$feat/s2vt/model_$epoch.pth \
# --batch_size 100 \
# --gpu 1 
