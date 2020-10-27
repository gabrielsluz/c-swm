'''
Script to evaluate fine tuning:
Trains a model and evaluates it every k epochs by fine tuning 
'''
import argparse
import torch
import utils
import datetime
import os
import pickle
import copy

import numpy as np
import logging

from torch.utils import data
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision

import modules

from collections import defaultdict
from fine_tune import *
import utils

'''
Returns the padded mnist dataset
'''
def load_padded_mnist(file_path):
    npzfile = np.load(file_path)
    padded_train_set_array = npzfile['arr_0']
    train_targets_array = npzfile['arr_1']
    padded_test_set_array = npzfile['arr_2']
    test_targets_array = npzfile['arr_3']

    return padded_train_set_array, train_targets_array, padded_test_set_array, test_targets_array

'''
Train model and evaluate every eval_every epochs
use_trans_model, ft_data_loader, ft_eval_data_loader => Evaluation parameters
'''
def train_and_eval(args, eval_every, use_trans_model, ft_data_loader, ft_eval_data_loader):
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    now = datetime.datetime.now()
    timestamp = now.isoformat()

    if args.name == 'none':
        exp_name = timestamp
    else:
        exp_name = args.name

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    exp_counter = 0
    save_folder = '{}/{}/'.format(args.save_folder, exp_name)

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    meta_file = os.path.join(save_folder, 'metadata.pkl')
    model_file = os.path.join(save_folder, 'model.pt')
    log_file = os.path.join(save_folder, 'log.txt')

    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger()
    logger.addHandler(logging.FileHandler(log_file, 'a'))
    print = logger.info

    pickle.dump({'args': args}, open(meta_file, "wb"))

    device = torch.device('cuda' if args.cuda else 'cpu')
    print("About to get dataset")
    transform = None
    if args.data_aug:
        transform = utils.get_data_augmentation()
    dataset = utils.StateTransitionsDataAugDataset(
        hdf5_file=args.dataset, transforms=transform)
    train_loader = data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    print("Dataset loaded")
    # Get data sample
    obs = train_loader.__iter__().next()[0]
    input_shape = obs[0].size()

    model = modules.ContrastiveSWM(
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        action_dim=args.action_dim,
        input_dims=input_shape,
        num_objects=args.num_objects,
        sigma=args.sigma,
        hinge=args.hinge,
        ignore_action=args.ignore_action,
        copy_action=args.copy_action,
        encoder=args.encoder,
        use_nt_xent_loss=args.use_nt_xent_loss,
        temperature=args.temperature).to(device)

    model.apply(utils.weights_init)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate)
    # Train model.
    print('Starting model training...')
    step = 0
    best_loss = 1e9

    epoch_acc_list = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0

        for batch_idx, data_batch in enumerate(train_loader):
            data_batch = [tensor.to(device) for tensor in data_batch]
            optimizer.zero_grad()

            if model.use_nt_xent_loss:
                loss = model.nt_xent_loss(*data_batch)
            else:
                loss = model.contrastive_loss(*data_batch)

            loss.backward()
            train_loss += loss.item()
            optimizer.step()

            if args.decoder:
                optimizer_dec.step()

            if batch_idx % args.log_interval == 0:
                print(
                    'Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data_batch[0]),
                        len(train_loader.dataset),
                        100. * batch_idx / len(train_loader),
                        loss.item() / len(data_batch[0])))

            step += 1

        avg_loss = train_loss / len(train_loader.dataset)
        print('====> Epoch: {} Average loss: {:.6f}'.format(
            epoch, avg_loss))

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), model_file)
        
        if epoch % eval_every == 0 or epoch == args.epochs:
            #Copy model for fine tuning
            model_clone = modules.ContrastiveSWM(
                embedding_dim=args.embedding_dim,
                hidden_dim=args.hidden_dim,
                action_dim=args.action_dim,
                input_dims=input_shape,
                num_objects=args.num_objects,
                sigma=args.sigma,
                hinge=args.hinge,
                ignore_action=args.ignore_action,
                copy_action=args.copy_action,
                encoder=args.encoder,
                use_nt_xent_loss=args.use_nt_xent_loss,
                temperature=args.temperature).to('cpu')
            model_clone.load_state_dict(copy.deepcopy(model.state_dict())) #Deepcopy does not work on model
            model_clone.to(device)
            ft_acc_list = fine_tune_and_eval_downstream(model_clone, device, ft_data_loader, ft_eval_data_loader,
             10, acc_every=6, use_trans_model=use_trans_model, epochs=60, learning_rate = 5e-4)
            model_clone.to('cpu')
            #Get best accuracy from list and use as the evaluation result for this training epoch
            best_ft_acc = max(ft_acc_list)
            epoch_acc_list.append((epoch, best_ft_acc))
            print('[Epoch %d] test acc: %.3f'%(epoch, best_ft_acc))

    return epoch_acc_list

'''
Main
'''

parser = argparse.ArgumentParser()
#Train
parser.add_argument('--batch-size', type=int, default=1024,
                    help='Batch size.')
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of training epochs.')
parser.add_argument('--learning-rate', type=float, default=5e-4,
                    help='Learning rate.')

parser.add_argument('--encoder', type=str, default='small',
                    help='Object extrator CNN size (e.g., `small`).')
parser.add_argument('--sigma', type=float, default=0.5,
                    help='Energy scale.')
parser.add_argument('--hinge', type=float, default=1.,
                    help='Hinge threshold parameter.')

parser.add_argument('--hidden-dim', type=int, default=512,
                    help='Number of hidden units in transition MLP.')
parser.add_argument('--embedding-dim', type=int, default=2,
                    help='Dimensionality of embedding.')
parser.add_argument('--action-dim', type=int, default=4,
                    help='Dimensionality of action space.')
parser.add_argument('--num-objects', type=int, default=5,
                    help='Number of object slots in model.')
parser.add_argument('--ignore-action', action='store_true', default=False,
                    help='Ignore action in GNN transition model.')
parser.add_argument('--copy-action', action='store_true', default=False,
                    help='Apply same action to all object slots.')

parser.add_argument('--decoder', action='store_true', default=False,
                    help='Train model using decoder and pixel-based loss.')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disable CUDA training.')
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed (default: 42).')
parser.add_argument('--log-interval', type=int, default=20,
                    help='How many batches to wait before logging'
                         'training status.')
parser.add_argument('--dataset', type=str,
                    default='data/shapes_train.h5',
                    help='Path to replay buffer.')
parser.add_argument('--name', type=str, default='none',
                    help='Experiment name.')
parser.add_argument('--save-folder', type=str,
                    default='checkpoints',
                    help='Path to checkpoints.')

parser.add_argument('--use-nt-xent-loss', action='store_true', default=False,
                    help='Uses SimCLR loss')
parser.add_argument('--temperature', type=float, default=0.1,
                    help='NT-Xent loss temperature')
parser.add_argument('--use-trans-model', action='store_true', default=False,
                    help='Use GNN in evaluation.')

parser.add_argument('--padded_mnist_path', type=str, default='data/padded_mnist.npz',
                    help='Path to padded_minist.npz')

parser.add_argument('--data-aug', action='store_true', default=False,
                    help='Use Data Augmentation')

#Results 
parser.add_argument('--eval-every', type=int, default=4,
                    help='Evaluate every k epochs')
parser.add_argument('--num-workers', type=int, default=4,
                    help='Number of data loaders')
parser.add_argument('--results-file', type=str,
                    default='results_c_swm.txt',
                    help='Path to file containing results')

args = parser.parse_args()

results_file = open(args.results_file, "a")

#Datasets for Fine Tuning
padded_train_set_array, train_targets_array, padded_test_set_array, test_targets_array = load_padded_mnist(args.padded_mnist_path)
#Fine tuning on 10%
mnist_ds = MNISTDataset(padded_train_set_array[:6000], train_targets_array[:6000])
mnist_data_loader = DataLoader(mnist_ds, batch_size=256, shuffle=True)
mnist_test_ds = MNISTDataset(padded_test_set_array, test_targets_array)
mnist_test_data_loader = DataLoader(mnist_test_ds, batch_size=5, shuffle=False)

#Train
epoch_acc_list = train_and_eval(args, args.eval_every, args.use_trans_model, mnist_data_loader, mnist_test_data_loader)
for epoch, acc in epoch_acc_list:
    args.epochs = epoch
    results_file.write(str(args.__dict__ ))
    results_file.write("\n----\n")
    results_file.write("use_trans_model=" + str(args.use_trans_model))
    results_file.write("\n----\n")
    results_file.write("FineTuning10pc = " + str(acc))
    results_file.write("\n----\n")
