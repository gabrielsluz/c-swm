'''
Unified script for doing linear evaluation and fine tuning
It can be used after train_and_eval.py 
'''
import argparse
import numpy as np
import torch
from torch.utils import data
import os
import pickle
import modules
import sklearn
from sklearn.linear_model import SGDClassifier
from linear_eval import generate_repr_dataset, linear_eval
from fine_tune import *

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
Loads model from files using c-swm format
'''
def load_model_from_file(save_folder, input_shape, device):
    meta_file = os.path.join(save_folder, 'metadata.pkl')
    model_file = os.path.join(save_folder, 'model.pt')
    train_args = pickle.load(open(meta_file, 'rb'))['args']
    model = modules.ContrastiveSWM(
        embedding_dim=train_args.embedding_dim,
        hidden_dim=train_args.hidden_dim,
        action_dim=train_args.action_dim,
        input_dims=input_shape,
        num_objects=train_args.num_objects,
        sigma=train_args.sigma,
        hinge=train_args.hinge,
        ignore_action=train_args.ignore_action,
        copy_action=train_args.copy_action,
        encoder=train_args.encoder).to(device)
    model.load_state_dict(torch.load(model_file))
    return model, train_args


'''
Main
'''
parser = argparse.ArgumentParser()
parser.add_argument('--padded_mnist_path', type=str, default='data/padded_mnist.npz',
                    help='Path to padded_minist.npz')
parser.add_argument('--save-folder', type=str,
                    default='checkpoints',
                    help='Path to checkpoints.')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disable CUDA training.')
            
#Results
parser.add_argument('--num-workers', type=int, default=4,
                    help='Number of data loaders')
parser.add_argument('--results-file', type=str,
                    default='results_c_swm.txt',
                    help='Path to file containing results')
                    
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
results_file = open(args.results_file, "a")

#Dataset
padded_train_set_array, train_targets_array, padded_test_set_array, test_targets_array = load_padded_mnist(args.padded_mnist_path)
train_loader = data.DataLoader(
    padded_train_set_array, batch_size=1024, shuffle=False, num_workers=args.num_workers)

test_loader = data.DataLoader(
    padded_test_set_array, batch_size=1024, shuffle=False, num_workers=args.num_workers)

#Model
obs = train_loader.__iter__().next()[0]
input_shape = obs.size()
device = torch.device('cuda' if args.cuda else 'cpu')

model, train_args = load_model_from_file(args.save_folder, input_shape, device)
results_file.write(str(train_args.__dict__ ))
results_file.write("\n----\n")

#Linear evaluation
train_repr_array = generate_repr_dataset(model, train_loader, device)
test_repr_array = generate_repr_dataset(model, test_loader, device)
eval_score = linear_eval(train_repr_array, train_targets_array, test_repr_array, test_targets_array)
results_file.write("LinearEvalAcc = " + str(eval_score))
results_file.write("\n----\n")
print("Linear eval acc = " + str(eval_score))

#Fine tuning on 10%
mnist_ds = MNISTDataset(padded_train_set_array[:6000], train_targets_array[:6000])
mnist_data_loader = DataLoader(mnist_ds, batch_size=256, shuffle=True)
mnist_test_ds = MNISTDataset(padded_test_set_array, test_targets_array)
mnist_test_data_loader = DataLoader(mnist_test_ds, batch_size=5, shuffle=False)

model, train_args = load_model_from_file(args.save_folder, input_shape, device) #Reload
model = fine_tune_downstream(model, device, mnist_data_loader, 10, epochs=30, learning_rate=5e-4)
fine_tune_acc = evaluate_downstream(model, device, mnist_test_data_loader)
results_file.write("FineTuning10pc30epochs = " + str(fine_tune_acc))
results_file.write("\n----\n")
print("Fine tune eval 30 acc = " + str(fine_tune_acc))

model = fine_tune_downstream(model, device, mnist_data_loader, 10, epochs=30, learning_rate=5e-4)
fine_tune_acc = evaluate_downstream(model, device, mnist_test_data_loader)
results_file.write("FineTuning10pc60epochs = " + str(fine_tune_acc))
results_file.write("\n----\n")
print("Fine tune eval 60 acc = " + str(fine_tune_acc))

results_file.close()