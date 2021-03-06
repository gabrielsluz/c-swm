#!/bin/bash
#num_objects => 2, 5, 10, 15, 20, 25
#Train10 => eval_all => eval_all_use_trans_model => eval 1 => eval 5 => eval 10
#Train20 => eval_all => eval_all_use_trans_model => eval 1 => eval 5 => eval 10

#for num_objs in 2 5 10 15 20 25; do
for num_objs in 15 20 25; do
  for num_epochs in 10 20; do
    python3 train.py --seed 1 --dataset /datasets/c_swm_data/mmnist_train.h5 --encoder large --embedding-dim 10 --num-objects $num_objs --epochs $num_epochs --name mmnist --ignore-action --batch-size 512
    python3 eval_all.py --padded_mnist_path /datasets/c_swm_data/padded_mnist.npz --save-folder checkpoints/mmnist/ --results-file eval_all_results.txt
    python3 eval_all.py --padded_mnist_path /datasets/c_swm_data/padded_mnist.npz --save-folder checkpoints/mmnist/ --results-file eval_all_results.txt --use-trans-model 
    python3 eval.py --dataset data/mmnist_eval.h5 --save-folder checkpoints/mmnist/ --num-steps 1 --results-file eval_all_results.txt
    python3 eval.py --dataset data/mmnist_eval.h5 --save-folder checkpoints/mmnist/ --num-steps 5 --results-file eval_all_results.txt
    python3 eval.py --dataset data/mmnist_eval.h5 --save-folder checkpoints/mmnist/ --num-steps 10 --results-file eval_all_results.txt
  done
done

