import shutil
import subprocess
import glob
import os
import numpy as np
import os
import argparse
import sys
from tqdm import tqdm
from PIL import Image
# caffe_root = 'your caffe root'
# sys.path.insert(0, caffe_root + '/python')
import caffe

def extract_frames(video, dst):
    with open(os.devnull, "w") as ffmpeg_log:
        if os.path.exists(dst):
            print(" cleanup: " + dst + "/")
            shutil.rmtree(dst)
        os.makedirs(dst)
        video_to_frames_command = ["ffmpeg",
                                   # (optional) overwrite output file if it exists
                                   '-y',
                                   '-i', video,  # input file
                                   '-vf', "scale=400:300",  # input file
                                   '-qscale:v', "2",  # quality for JPEG
                                   '{0}/%06d.jpg'.format(dst)]
        subprocess.call(video_to_frames_command,
                        stdout=ffmpeg_log, stderr=ffmpeg_log)


def extract_feats(params, net):
    dir_fc = params['output_dir']
    if not os.path.isdir(dir_fc):
        os.mkdir(dir_fc)
    print("save video feats to %s" % (dir_fc))
    video_list = glob.glob(os.path.join(params['video_path'], '*.mp4'))
    mean_value = np.array([104.00698793, 116.66876762, 122.67891434])
    # np.array((102.144, 102.144, 108.64))
    for video in tqdm(video_list):
        video_id = video.split("/")[-1].split(".")[0]
        dst = video_id
        extract_frames(video, dst)
        image_list = sorted(glob.glob(os.path.join(dst, '*.jpg')))
        samples = np.round(np.linspace(
            0, len(image_list) - 1, params['n_frame_steps']))
        image_list = [image_list[int(sample)] for sample in samples]
        ims = []
        img_feats = []
        for index, iImg in enumerate(range(len(image_list))):
            im = Image.open(image_list[iImg])
            im = im.resize((224, 224), Image.BILINEAR)
            im = np.array(im, dtype=np.float32)
            im = im[:, :, ::-1]  # RGB->BGR
            im -= mean_value  # BGR
            im = im.transpose((2, 0, 1))  # im:(c,h,w)
            im = im[np.newaxis, ...]
            ims.append(im)
            if (index+1) % params['batch_size'] == 0:
                ims = np.concatenate(ims, axis=0)
                net.blobs['data'].reshape(*ims.shape)
                net.blobs['data'].data[...] = ims
                output = net.forward()
                img_feats.append(net.blobs['pool5'].data.squeeze())
                ims = []
        img_feats = np.concatenate(img_feats, axis=0)
        # Save the inception features
        outfile = os.path.join(dir_fc, video_id + '.npy')
        np.save(outfile, img_feats)
        # cleanup
        shutil.rmtree(dst)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", dest='gpu', type=int, default=0,
                        help='Set CUDA_VISIBLE_DEVICES environment variable, optional')
    parser.add_argument('--batch_size', type=int, default=20, help='minibatch size')
    parser.add_argument("--output_dir", dest='output_dir', type=str,
                        default='data/feats/resnet152', help='directory to store features')
    parser.add_argument("--n_frame_steps", dest='n_frame_steps', type=int, default=80,
                        help='how many frames to sampler per video')
    parser.add_argument("--video_path", dest='video_path', type=str,
                        default='data/videos', help='path to video dataset')
    parser.add_argument("--model_weight", dest="model_weight", type=str,
                        default='pretrained_models/resnet152_places365.caffemodel',
                        help='model_weight')
    parser.add_argument("--model_deploy", dest="model_deploy", type=str,
                        default='pretrained_models/deploy_resnet152_places365.prototxt',
                        help='deploy')
    args = parser.parse_args()
    params = vars(args)
    # TODO: remove this limit
    assert params['n_frame_steps'] % params['batch_size'] == 0, 'For simplicity, n_frame_steps%batch_size must = 0'
    caffe.set_device(params['gpu'])
    caffe.set_mode_gpu()
    model_weights = params['model_weight']
    model_def = params['model_deploy']
    net = caffe.Net(model_def, model_weights, caffe.TEST)
    extract_feats(params, net)
