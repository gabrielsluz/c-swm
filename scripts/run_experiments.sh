#!/bin/bash
#num_epochs

for seed in 1 2 3 ; do
  python3 train_eval_fine_tune.py --seed $seed --dataset /datasets/c_swm_data/mmnist_train.h5 --encoder large --embedding-dim 10 --num-objects 15 --epochs 20 --name mmnist --ignore-action --batch-size 512 --padded_mnist_path /datasets/c_swm_data/padded_mnist.npz
  python3 train_eval_fine_tune.py --seed $seed --dataset /datasets/c_swm_data/mmnist_train.h5 --encoder large --embedding-dim 10 --num-objects 15 --epochs 20 --name mmnist --ignore-action --batch-size 512 --use-nt-xent-loss --temperature 0.1 --padded_mnist_path /datasets/c_swm_data/padded_mnist.npz
  python3 train_eval_fine_tune.py --seed $seed --dataset /datasets/c_swm_data/mmnist_train.h5 --encoder large --embedding-dim 10 --num-objects 15 --epochs 20 --name mmnist --ignore-action --batch-size 512 --use-nt-xent-loss --temperature 0.1 --padded_mnist_path /datasets/c_swm_data/padded_mnist.npz --use-trans-model
done
