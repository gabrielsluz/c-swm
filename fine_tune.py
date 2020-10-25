'''
Functions for fine tuning the model
'''
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import numpy as np

class MNISTDataset(Dataset):
    def __init__(self, images, labels, transforms=None):
        self.X = images
        self.y = labels
        self.transforms = transforms
         
    def __len__(self):
        return (self.X.shape[0])
    
    def __getitem__(self, i):
        data = self.X[i]
        if self.transforms:
            data = self.transforms(data)    
        return (data, self.y[i])

"""
Fine tunes the model with a softmax head on a downstream classification task
"""
def fine_tune_downstream(model, device, data_loader,  num_classes, use_trans_model=False, epochs=20, learning_rate = 5e-5):
    num_ftrs = model.num_objects * model.embedding_dim
    model.class_head = nn.Linear(num_ftrs, num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    print('Start tuning model')
    model.train()
    for epoch in range(epochs):
        tuning_loss = 0
        for data in data_loader:
            obs = data[0].type(torch.FloatTensor)
            obs = obs.to(device)
            targets = data[1].to(device)
            optimizer.zero_grad()
            
            state = model.obj_encoder(model.obj_extractor(obs))
            if use_trans_model:
                pred_trans = model.transition_model(state, np.zeros((1,1), dtype=np.int64))
                state = state + pred_trans
            state = torch.flatten(state, start_dim=1)
            outputs = model.class_head(state)
            
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
 
            tuning_loss += loss.item()
        print('[Epoch %d] loss: %.3f' %
                      (epoch + 1, tuning_loss/len(data_loader)))
 
    print('Done Tuning')
    return model


"""
Evaluates the model on a downstream classification task
"""
def evaluate_downstream(model, device, data_loader, use_trans_model=False):
    model.eval()
    hits = 0.0
    with torch.no_grad():
        for data in data_loader:
            obs = data[0].type(torch.FloatTensor)
            obs = obs.to(device)
            targets = data[1].to(device)
            
            state = model.obj_encoder(model.obj_extractor(obs))
            if use_trans_model:
                pred_trans = model.transition_model(state, np.zeros((1,1), dtype=np.int64))
                state = state + pred_trans
            state = torch.flatten(state, start_dim=1)
            outputs = model.class_head(state)
            pred = outputs.data.max(1, keepdim=True)[1]
            hits += pred.eq(targets.data.view_as(pred)).sum()
    hits = hits.item()
    acc = hits / len(data_loader.dataset)
    return acc


'''
A function for fine tuning and evaluating every k epochs
Returns a list of accuracy values
'''
def fine_tune_and_eval_downstream(model, device, data_loader, eval_data_loader, num_classes, acc_every=5, use_trans_model=False, epochs=20, learning_rate = 5e-5):
    num_ftrs = model.num_objects * model.embedding_dim
    model.class_head = nn.Linear(num_ftrs, num_classes).to(device)
    epoch_acc_list = []

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    print('Start tuning model')
    model.train()
    for epoch in range(epochs):
        tuning_loss = 0
        for data in data_loader:
            obs = data[0].type(torch.FloatTensor)
            obs = obs.to(device)
            targets = data[1].to(device)
            optimizer.zero_grad()
            
            state = model.obj_encoder(model.obj_extractor(obs))
            if use_trans_model:
                pred_trans = model.transition_model(state, np.zeros((1,1), dtype=np.int64))
                state = state + pred_trans
            state = torch.flatten(state, start_dim=1)
            outputs = model.class_head(state)
            
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
 
            tuning_loss += loss.item()
        print('[Epoch %d] loss: %.3f' %
                      (epoch + 1, tuning_loss/len(data_loader)))
        if epoch % acc_every == 0 or epoch == epochs-1:
            acc = evaluate_downstream(model, device, eval_data_loader, use_trans_model)
            #epoch_acc_list.append((epoch+1, acc))
            epoch_acc_list.append(acc)
            print('[Epoch %d] test acc: %.3f'%(epoch+1, acc))
 
    print('Done Tuning')
    return epoch_acc_list