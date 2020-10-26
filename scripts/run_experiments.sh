#!/bin/bash

for seed in 5 6 7; do
  python3 train_eval_fine_tune.py --seed $seed --dataset /datasets/c_swm_data/mmnist_train.h5 --encoder large --embedding-dim 10 --num-objects 15 --epochs 12 --name mmnist --ignore-action --batch-size 512 --padded_mnist_path /datasets/c_swm_data/padded_mnist.npz --results-file eval_all_results.txt --use-nt-xent-loss --temperature 0.1 --eval-every 3
  python3 train_eval_fine_tune.py --seed $seed --dataset /datasets/c_swm_data/mmnist_train.h5 --encoder large --embedding-dim 10 --num-objects 15 --epochs 12 --name mmnist --ignore-action --batch-size 512 --padded_mnist_path /datasets/c_swm_data/padded_mnist.npz --results-file eval_all_results.txt --use-nt-xent-loss --temperature 0.1 --use-trans-model --eval-every 3

  python3 train_eval_fine_tune.py --seed $seed --dataset /datasets/c_swm_data/mmnist_train.h5 --encoder large --embedding-dim 10 --num-objects 15 --epochs 12 --name mmnist --ignore-action --batch-size 512 --padded_mnist_path /datasets/c_swm_data/padded_mnist.npz --results-file eval_all_results.txt --use-nt-xent-loss --temperature 0.5 --eval-every 3
  python3 train_eval_fine_tune.py --seed $seed --dataset /datasets/c_swm_data/mmnist_train.h5 --encoder large --embedding-dim 10 --num-objects 15 --epochs 12 --name mmnist --ignore-action --batch-size 512 --padded_mnist_path /datasets/c_swm_data/padded_mnist.npz --results-file eval_all_results.txt --use-nt-xent-loss --temperature 0.5 --use-trans-model --eval-every 3

  python3 train_eval_fine_tune.py --seed $seed --dataset /datasets/c_swm_data/mmnist_train.h5 --encoder large --embedding-dim 10 --num-objects 15 --epochs 12 --name mmnist --ignore-action --batch-size 512 --padded_mnist_path /datasets/c_swm_data/padded_mnist.npz --results-file eval_all_results.txt --eval-every 3
  python3 train_eval_fine_tune.py --seed $seed --dataset /datasets/c_swm_data/mmnist_train.h5 --encoder large --embedding-dim 10 --num-objects 15 --epochs 12 --name mmnist --ignore-action --batch-size 512 --padded_mnist_path /datasets/c_swm_data/padded_mnist.npz --results-file eval_all_results.txt --use-trans-model --eval-every 3
done
