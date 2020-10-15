'''
Transforms the mnist_test_seq.npy from http://www.cs.toronto.edu/~nitish/unsupervised_video/
into the format expected by C-SWM.
Divides into training and validation sets.
Format:
observations, actions, next_observations
Each on is a list
'''
# Get env directory
import sys
from pathlib import Path
if str(Path.cwd()) not in sys.path:
    sys.path.insert(0, str(Path.cwd()))

import numpy as np
from utils import save_list_dict_h5py

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mmnist_npy_path', type=str,
                    default='mnist_test_seq.npy',
                    help='File name / path.')
parser.add_argument('--train_fname', type=str,
                    default='data/mmnist_train.h5',
                    help='File name / path.')
parser.add_argument('--eval_fname', type=str,
                    default='data/mmnist_eval.h5',
                    help='File name / path.')
parser.add_argument('--num-frames', type=int, default=20,
                    help='Number of frames to use.')
parser.add_argument('--train_share', type=float, default=0.8,
                    help='Percentage of the dataset as training')

args = parser.parse_args()


mmnist_npy_path = args.mmnist_npy_path
mmnist_dataset = np.load(mmnist_npy_path)

num_videos = mmnist_dataset.shape[1]
num_frames = args.num_frames

num_videos_train = int(args.train_share * num_videos)

dataset_train = []
dataset_eval = []

for video_index in range(num_videos_train):
    sample = {
        'obs': [],
        'next_obs': [],
        'action': []
    }
    for frame_index in range(num_frames - 1):
        sample['obs'].append(mmnist_dataset[frame_index, video_index, :,:])
        sample['next_obs'].append(mmnist_dataset[frame_index+1, video_index, :,:])
        sample['action'].append(np.zeros((1,1)))
    dataset_train.append(sample)
        
for video_index in range(num_videos_train, num_videos):
    sample = {
        'obs': [],
        'next_obs': [],
        'action': []
    }
    for frame_index in range(num_frames - 1):
        sample['obs'].append(mmnist_dataset[frame_index, video_index, :,:])
        sample['next_obs'].append(mmnist_dataset[frame_index+1, video_index, :,:])
        sample['action'].append(np.zeros((1,1)))
    dataset_eval.append(sample)
    
save_list_dict_h5py(dataset_train, args.train_fname)
save_list_dict_h5py(dataset_eval, args.eval_fname)

