# requirements #

- cuda
- pytorch 0.3.1
- python3(未测试) or python2(已测试,最好统一用py2吧)
- ffmpeg (can install using anaconda)

# usage #

1. 2d特征提取, 如resnet101, nasnet等
```bash
sh ./2d_extract_feat.sh
# model 模型选择
# n_frame_steps 一段视频提取多少帧，默认选80吧
```

2. 3d特征提取
```bash
cd c3d_feat_extract
sh ./c3d_feat_extract.sh
# --mode feature 提取特征模式，无需改动
# 以下根据所选模型不同进行更改
# --model_name resnext \
# --model_depth 101 \
# --resnext_cardinality 32 \
# --resnet_shortcut B \
# --model pretrained_models/resnext-101-64f-kinetics.pth
```
3. 训练

```bash
./train_s2vt.sh
# 根据相关配置进行设置，具体选项含义参考opts.py
```

4. 测试和评分

```bash
./eval_s2vt.sh
# 根据相关配置进行设置，具体选项含义参考eval.py
```
# file tree #

相关文件下载 
链接: https://pan.baidu.com/s/1RDNygrWtz_PtVH8nh4vG3w 密码: nxyk
```
data
│   all_caption.json
│   all_info.json    
│   all_videodatainfo_2017.json
└───feats
│   └───nasnet
│   │   │   videoxxx.npy
│   │   │   ...
│   └───resnet
│   │   │   videoxxx.npy
│   │   │   ... 
│   └───xxnet
│       │   videoxxx.npy
│       │   ... 
└───videos
│   │   videoxxx.mp4
│   │   ...
│
│
新建这些目录
log
checkpoint
result

```

# pytorch implementation of video captioning

recommend installing pytorch and python packages using Anaconda


### python packages

- tqdm
- pillow
- pretrainedmodels
- nltk

## Data

MSR-VTT. Test video doesn't have captions, so I spilit train-viedo to train/val/test. Extract and put them in `./data/` directory

- train-video: [download link](https://drive.google.com/file/d/1Qi6Gn_l93SzrvmKQQu-drI90L-x8B0ly/view?usp=sharing)
- test-video: [download link](https://drive.google.com/file/d/10fPbEhD-ENVQihrRvKFvxcMzkDlhvf4Q/view?usp=sharing)
- json info of train-video: [download link](https://drive.google.com/file/d/1LcTtsAvfnHhUfHMiI4YkDgN7lF1-_-m7/view?usp=sharing)
- json info of test-video: [download link](https://drive.google.com/file/d/1Kgra0uMKDQssclNZXRLfbj9UQgBv-1YE/view?usp=sharing)

## Options

all default options are defined in opt.py or corresponding code file, change them for your like.

## Usage

### (Optional) c3d features
you can use [video-classification-3d-cnn-pytorch](https://github.com/kenshohara/video-classification-3d-cnn-pytorch) to extract features from video. Then mean pool to get a 2048 dim feature for each video.

### Steps

1. preprocess videos and labels

    this steps take about 3 hours for msr-vtt datasets use one titan XP gpu

```bash
python prepro_feats.py --output_dir data/feats/resnet152 --model resnet152 --n_frame_steps 40  --gpu 4,5

python prepro_vocab.py
```

2. Training a model

```bash

python train.py --gpu 5,6,7 --epochs 9001 --batch_size 450 --checkpoint_path data/save --feats_dir data/feats/resnet152 --dim_vid 2048 --model S2VTAttModel
```

3. test

    opt_info.json will be in same directory as saved model.

```bash
python eval.py --recover_opt data/save/opt_info.json --saved_model data/save/model_1000.pth --batch_size 100 --gpu 1,0
```

## Metrics

I fork the [coco-caption XgDuan](https://github.com/XgDuan/coco-caption/tree/python3). Thanks to port it to python3.

## TODO
- lstm
- beam search
- reinforcement learning

## Note
This repository is not maintained, please see my another repository [video-caption-openNMT.py](https://github.com/xiadingZ/video-caption-openNMT.pytorch). It has higher performence and test score.
