import pandas as pd
import numpy as np
import matplotlib as mpl

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import random
import higher

class StateDataset(torch.utils.data.Dataset):
    def __init__(self, epoch_len=5, x_range=(-15,15),batch_size=5):

        self.epoch_len = epoch_len
        self.x_range = x_range
        self.batch_size = batch_size

    def __len__(self):
        return self.epoch_len

    def __getitem__(self, idx):
        X = np.random.uniform(self.x_range[0], self.x_range[1], (self.batch_size,2))
        return X

class MLP(torch.nn.Module):
        def __init__(self, input_size, hidden_size):
            super(MLP, self).__init__()
            self.input_size = input_size
            self.hidden_size  = hidden_size
            self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
            self.relu = torch.nn.ReLU()
            self.fc2 = torch.nn.Linear(self.hidden_size, self.hidden_size)
            self.fc3 = torch.nn.Linear(self.hidden_size, 1)
            self.sigmoid = torch.nn.Sigmoid()
        
        def forward(self, x):
            hidden1 = self.fc1(x)
            relu1 = self.relu(hidden1)
            hidden2 = self.fc2(relu1)
            relu2 = self.relu(hidden2)
            output = self.fc3(relu2)
            return output

def mpc_loss(x_list, u_list, A, B, Q, R):
    loss=0
    for i in range(len(u_list)):
        loss += (x_list[i+1]@Q@x_list[i+1].T + u_list[i]@R@u_list[i].T)#**0.5
    return loss

def main(
        batch_size=1,
        opt_steps=5,
        meta_lr=0.003,
        fast_lr=0.0005,
        meta_batch_size=256,
        adaptation_steps=1,
        epoch_len=128,
        epochs=1,
        cuda=True,
        seed=58):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device('cpu')
    if cuda and torch.cuda.device_count():
        torch.cuda.manual_seed(seed)
        device = torch.device('cuda')

    dataset = StateDataset(epoch_len=epoch_len, batch_size=meta_batch_size)
    training_generator = torch.utils.data.DataLoader(dataset)
    
    model = MLP(2, 256).to(device)
    meta_opt = optim.Adam(model.parameters(), lr=1e-3)
    A = torch.tensor(np.array([[1.0,1.0],[0.0,1.0]])).to(device).float()
    B = torch.tensor(np.array([[0.0],[1.0]])).to(device).float()
    Q = torch.tensor(0.01*np.diag([1.0, 1.0])).to(device).float()
    R = torch.tensor([[0.1]]).to(device).float()

    epochs = 0

    for X in training_generator:
        inner_opt = optim.SGD(model.parameters(), lr=1e-5)
        meta_opt.zero_grad()
        meta_train_loss = 0.0
        X = X.to(device).float()

        for j in range(meta_batch_size):
            with higher.innerloop_ctx(
                model, inner_opt, copy_initial_weights=False
            ) as (fnet, diffopt):
                x = X[0,j:j+1]

                for k in range(1):
                    T = 10
                    x_list = []
                    x_list.append(x)
                    u_list = []
                    for i in range(T):
                        u = fnet(x_list[i])
                        x_next = A@x_list[i].T + B@u.T
                        x_list.append(x_next.T)
                        u_list.append(u)
                    inner_loss = mpc_loss(x_list, u_list, A, B, Q, R)
                    diffopt.step(inner_loss)
                    #print(inner_loss.item())
                    #print(x_list, u_list)

                T = 10
                x_list = []
                x_list.append(x)
                u_list = []
                for i in range(T):
                    u = fnet(x_list[i])
                    x_next = A@x_list[i].T + B@u.T
                    x_list.append(x_next.T)
                    u_list.append(u)
                meta_loss = mpc_loss(x_list, u_list, A, B, Q, R)/meta_batch_size
                meta_train_loss += meta_loss.item()
                meta_loss.backward()
                if(epochs%10==0 and j==0):
                    print('***EPOCH: '+str(epochs)+'***')
                    for n in range(len(u_list)):
                        print('X' + str(n) + ': ' , x_list[n].tolist())
                        print('U' + str(n) + ': ' , u_list[n].item())
                    print('******')
        epochs += 1
        print(meta_train_loss)
        meta_opt.step()

    torch.save({'model' : model.state_dict(), 'opt_meta' : meta_opt.state_dict()}, 'maml_weights/new_mamlsgd')


if __name__=='__main__':
    main()