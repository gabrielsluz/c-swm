#!/bin/bash

python3 train_eval_fine_tune.py --seed 8 --dataset /datasets/c_swm_data/mmnist_full_train.h5 --encoder large --embedding-dim 10 --num-objects 15 --epochs 20 --name mmnist --ignore-action --batch-size 512 --padded_mnist_path /datasets/c_swm_data/padded_mnist.npz --results-file eval_all_results.txt --use-nt-xent-loss --temperature 0.1 --use-trans-model --eval-every 2 --data-aug
python3 train_eval_fine_tune.py --seed 8 --dataset /datasets/c_swm_data/mmnist_full_train.h5 --encoder large --embedding-dim 10 --num-objects 15 --epochs 20 --name mmnist --ignore-action --batch-size 512 --padded_mnist_path /datasets/c_swm_data/padded_mnist.npz --results-file eval_all_results.txt --use-nt-xent-loss --temperature 0.1 --use-trans-model --eval-every 2
python3 train_eval_fine_tune.py --seed 9 --dataset /datasets/c_swm_data/mmnist_full_train.h5 --encoder large --embedding-dim 10 --num-objects 15 --epochs 20 --name mmnist --ignore-action --batch-size 512 --padded_mnist_path /datasets/c_swm_data/padded_mnist.npz --results-file eval_all_results.txt --use-nt-xent-loss --temperature 0.1 --use-trans-model --eval-every 2 --data-aug
python3 train_eval_fine_tune.py --seed 9 --dataset /datasets/c_swm_data/mmnist_full_train.h5 --encoder large --embedding-dim 10 --num-objects 15 --epochs 20 --name mmnist --ignore-action --batch-size 512 --padded_mnist_path /datasets/c_swm_data/padded_mnist.npz --results-file eval_all_results.txt --use-nt-xent-loss --temperature 0.1 --use-trans-model --eval-every 2


for seed in 5 6 7 9; do
  python3 train_eval_fine_tune.py --seed $seed --dataset /datasets/c_swm_data/mmnist_full_train.h5 --encoder large --embedding-dim 10 --num-objects 15 --epochs 20 --name mmnist --ignore-action --batch-size 512 --padded_mnist_path /datasets/c_swm_data/padded_mnist.npz --results-file eval_all_results.txt --use-nt-xent-loss --temperature 0.1 --use-trans-model --eval-every 2 --data-aug --slot-attn
done

