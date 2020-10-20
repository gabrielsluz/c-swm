'''
Contains the functions used for linear protocol evaluation
'''
import torch
import numpy as np
import sklearn
from sklearn.linear_model import SGDClassifier

'''
Generates new dataset by applying model to modified MNIST data loader
Return the generated numpy array
'''
def generate_repr_dataset(model, data_loader, device, use_trans_model=False):
    repr_train = None
    with torch.no_grad():
        for batch_idx, data_batch in enumerate(data_loader):
            obs = data_batch.type(torch.FloatTensor)
            obs = obs.to(device)
            state = model.obj_encoder(model.obj_extractor(obs))
            if use_trans_model:
                pred_trans = model.transition_model(state, np.zeros((1,1), dtype=np.int64))
                state = state + pred_trans
            representation = torch.flatten(state, start_dim=1).cpu().numpy()
            if repr_train is None:
                repr_train = representation
            else:
                repr_train = np.concatenate((repr_train, representation))
    return repr_train

'''
Evaluates model by using it as a feature extractor for a linear classifier
'''
def linear_eval(train_repr_array, train_targets_array, test_repr_array, test_targets_array):
    linear_model = SGDClassifier(max_iter=2000, tol=1e-5)
    linear_model.fit(train_repr_array, train_targets_array)
    return linear_model.score(test_repr_array, test_targets_array)