python3 train.py --dataset /datasets/c_swm_data/mmnist_train.h5 --encoder large --embedding-dim 10 --num-objects 2 --epochs 50 --name mmnist --ignore-action

python3 eval_all.py --padded_mnist_path /datasets/c_swm_data/padded_mnist.npz --save-folder checkpoints/mmnist/ --results-file eval_all_results.txt

python3 train.py --dataset /datasets/c_swm_data/mmnist_train.h5 --encoder large --embedding-dim 20 --num-objects 2 --epochs 50 --name mmnist --ignore-action

python3 eval_all.py --padded_mnist_path /datasets/c_swm_data/padded_mnist.npz --save-folder checkpoints/mmnist/ --results-file eval_all_results.txt

python3 train.py --dataset /datasets/c_swm_data/mmnist_train.h5 --encoder large --embedding-dim 2 --num-objects 10 --epochs 50 --name mmnist --ignore-action

python3 eval_all.py --padded_mnist_path /datasets/c_swm_data/padded_mnist.npz --save-folder checkpoints/mmnist/ --results-file eval_all_results.txt

python3 train.py --dataset /datasets/c_swm_data/mmnist_train.h5 --encoder large --embedding-dim 2 --num-objects 20 --epochs 50 --name mmnist --ignore-action

python3 eval_all.py --padded_mnist_path /datasets/c_swm_data/padded_mnist.npz --save-folder checkpoints/mmnist/ --results-file eval_all_results.txt
