import argparse
import torch
import utils
import datetime
import os
import pickle

import numpy as np
import logging

from torch.utils import data
import torch.nn.functional as F

import modules

from collections import defaultdict

'''
Train model
'''
def train_c_swm(args):
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    print("Inside train_c_swm")

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
    dataset = utils.StateTransitionsDataset(
        hdf5_file=args.dataset)
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
        encoder=args.encoder).to(device)

    model.apply(utils.weights_init)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate)

    if args.decoder:
        if args.encoder == 'large':
            decoder = modules.DecoderCNNLarge(
                input_dim=args.embedding_dim,
                num_objects=args.num_objects,
                hidden_dim=args.hidden_dim // 16,
                output_size=input_shape).to(device)
        elif args.encoder == 'medium':
            decoder = modules.DecoderCNNMedium(
                input_dim=args.embedding_dim,
                num_objects=args.num_objects,
                hidden_dim=args.hidden_dim // 16,
                output_size=input_shape).to(device)
        elif args.encoder == 'small':
            decoder = modules.DecoderCNNSmall(
                input_dim=args.embedding_dim,
                num_objects=args.num_objects,
                hidden_dim=args.hidden_dim // 16,
                output_size=input_shape).to(device)
        decoder.apply(utils.weights_init)
        optimizer_dec = torch.optim.Adam(
            decoder.parameters(),
            lr=args.learning_rate)


    # Train model.
    print('Starting model training...')
    step = 0
    best_loss = 1e9

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0

        for batch_idx, data_batch in enumerate(train_loader):
            data_batch = [tensor.to(device) for tensor in data_batch]
            optimizer.zero_grad()

            if args.decoder:
                optimizer_dec.zero_grad()
                obs, action, next_obs = data_batch
                objs = model.obj_extractor(obs)
                state = model.obj_encoder(objs)

                rec = torch.sigmoid(decoder(state))
                loss = F.binary_cross_entropy(
                    rec, obs, reduction='sum') / obs.size(0)

                next_state_pred = state + model.transition_model(state, action)
                next_rec = torch.sigmoid(decoder(next_state_pred))
                next_loss = F.binary_cross_entropy(
                    next_rec, next_obs,
                    reduction='sum') / obs.size(0)
                loss += next_loss
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
        # print('====> Epoch: {} Average loss: {:.6f}'.format(
        #     epoch, avg_loss))

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), model_file)

    return model
'''
Eval
'''
def eval_c_swm(args, model):
    torch.backends.cudnn.deterministic = True

    # meta_file = os.path.join(args.save_folder, 'metadata.pkl')
    # model_file = os.path.join(args.save_folder, 'model.pt')

    # args = pickle.load(open(meta_file, 'rb'))['args']

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.batch_size = 100
    args.dataset = args.dataset_eval
    args.seed = 0

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    device = torch.device('cuda' if args.cuda else 'cpu')

    dataset = utils.PathDataset(
        hdf5_file=args.dataset, path_length=args.num_steps)
    eval_loader = data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Get data sample
    obs = eval_loader.__iter__().next()[0]
    input_shape = obs[0][0].size()

    # model = modules.ContrastiveSWM(
    #     embedding_dim=args.embedding_dim,
    #     hidden_dim=args.hidden_dim,
    #     action_dim=args.action_dim,
    #     input_dims=input_shape,
    #     num_objects=args.num_objects,
    #     sigma=args.sigma,
    #     hinge=args.hinge,
    #     ignore_action=args.ignore_action,
    #     copy_action=args.copy_action,
    #     encoder=args.encoder).to(device)

    # model.load_state_dict(torch.load(model_file))
    model.eval()

    # topk = [1, 5, 10]
    topk = [1]
    hits_at = defaultdict(int)
    num_samples = 0
    rr_sum = 0

    pred_states = []
    next_states = []

    with torch.no_grad():

        for batch_idx, data_batch in enumerate(eval_loader):
            data_batch = [[t.to(
                device) for t in tensor] for tensor in data_batch]
            observations, actions = data_batch

            if observations[0].size(0) != args.batch_size:
                continue

            obs = observations[0]
            next_obs = observations[-1]

            state = model.obj_encoder(model.obj_extractor(obs))
            next_state = model.obj_encoder(model.obj_extractor(next_obs))

            pred_state = state
            for i in range(args.num_steps):
                pred_trans = model.transition_model(pred_state, actions[i])
                pred_state = pred_state + pred_trans

            pred_states.append(pred_state.cpu())
            next_states.append(next_state.cpu())

        pred_state_cat = torch.cat(pred_states, dim=0)
        next_state_cat = torch.cat(next_states, dim=0)

        full_size = pred_state_cat.size(0)

        # Flatten object/feature dimensions
        next_state_flat = next_state_cat.view(full_size, -1)
        pred_state_flat = pred_state_cat.view(full_size, -1)

        dist_matrix = utils.pairwise_distance_matrix(
            next_state_flat, pred_state_flat)
        dist_matrix_diag = torch.diag(dist_matrix).unsqueeze(-1)
        dist_matrix_augmented = torch.cat(
            [dist_matrix_diag, dist_matrix], dim=1)

        # Workaround to get a stable sort in numpy.
        dist_np = dist_matrix_augmented.numpy()
        indices = []
        for row in dist_np:
            keys = (np.arange(len(row)), row)
            indices.append(np.lexsort(keys))
        indices = np.stack(indices, axis=0)
        indices = torch.from_numpy(indices).long()

        print('Processed {} batches of size {}'.format(
            batch_idx + 1, args.batch_size))

        labels = torch.zeros(
            indices.size(0), device=indices.device,
            dtype=torch.int64).unsqueeze(-1)

        num_samples += full_size
        print('Size of current topk evaluation batch: {}'.format(
            full_size))

        for k in topk:
            match = indices[:, :k] == labels
            num_matches = match.sum()
            hits_at[k] += num_matches.item()

        match = indices == labels
        _, ranks = match.max(1)

        reciprocal_ranks = torch.reciprocal(ranks.double() + 1)
        rr_sum += reciprocal_ranks.sum()

        pred_states = []
        next_states = []

    for k in topk:
        print('Hits @ {}: {}'.format(k, hits_at[k] / float(num_samples)))

    print('MRR: {}'.format(rr_sum / float(num_samples)))
    results_dict = {}
    results_dict['H1'] = hits_at[1] / float(num_samples)
    results_dict['MRR'] = rr_sum / float(num_samples)
    return results_dict


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

#Eval
parser.add_argument('--num-steps', type=int, default=1,
                    help='Number of prediction steps to evaluate.')
parser.add_argument('--dataset-eval', type=str,
                    default='data/shapes_eval.h5',
                    help='Dataset string.')

#Results 
parser.add_argument('--num-reps', type=int, default=1,
                    help='Number of times to evaluate')
parser.add_argument('--num-workers', type=int, default=4,
                    help='Number of data loaders')
parser.add_argument('--results-file', type=str,
                    default='results_c_swm.txt',
                    help='Path to file containing results')

args = parser.parse_args()

results_file = open(args.results_file, "a")
for i in range(args.num_reps):
    print("Repetition " + str(i))

    model = train_c_swm(args)
    results_file.write(str(args.__dict__ ))
    results_file.write("\n----\n")
    print(args.__dict__)

    results_dict = eval_c_swm(args, model)
    results_file.write(str(results_dict))
    results_file.write("\n----\n")
    print(results_dict)
    