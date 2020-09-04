import pandas as pd
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import random

SEED = 47

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device('cuda')

A = torch.tensor(np.array([[1.0,1.0],[0.0,1.0]])).to(device).double()
B = torch.tensor(np.array([[0.0],[1.0]])).to(device).double()
Q = torch.tensor(0.01*np.diag([1.0, 1.0])).to(device).double()
R = torch.tensor([[0.1]]).to(device).double()

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
        def __init__(self, hidden_size, state_len=2, control_len=1, T=5):
            super(MLP, self).__init__()
            self.state_len = state_len
            self.control_len = control_len
            self.T = T
            self.input_size = state_len + control_len*3*T
            self.hidden_size  = hidden_size
            self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
            self.relu = torch.nn.ReLU()
            self.fc2 = torch.nn.Linear(self.hidden_size, self.hidden_size)
            self.fc3 = torch.nn.Linear(self.hidden_size, T)
            self.gate = torch.nn.Linear(self.input_size, T)
            self.sigmoid = torch.nn.Sigmoid()
        
        def forward(self, x):
            hidden1 = self.fc1(x)
            relu1 = self.relu(hidden1)
            hidden2 = self.fc2(relu1)
            relu2 = self.relu(hidden2)
            grad_update = self.fc3(relu2)
            gate = self.sigmoid(self.gate(x))
            #print(gate)
            output = gate*x[:,self.state_len:self.state_len+self.control_len*self.T] + (1-gate)*grad_update

            return output

class MLP2(torch.nn.Module):
        def __init__(self, hidden_size, state_len=2, control_len=1, T=5):
            super(MLP2, self).__init__()
            self.state_len = state_len
            self.control_len = control_len
            self.T = T
            self.input_size = state_len
            self.hidden_size  = hidden_size
            self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
            self.relu = torch.nn.ReLU()
            self.fc2 = torch.nn.Linear(self.hidden_size, self.hidden_size)
            self.fc3 = torch.nn.Linear(self.hidden_size, T)
            self.gate = torch.nn.Linear(self.input_size, T)
            self.sigmoid = torch.nn.Sigmoid()
        
        def forward(self, x):
            hidden1 = self.fc1(x)
            relu1 = self.relu(hidden1)
            hidden2 = self.fc2(relu1)
            relu2 = self.relu(hidden2)
            grad_update = self.fc3(relu2)
            #gate = self.sigmoid(self.gate(x))
            #print(gate)
            #output = gate*x[:,self.state_len:self.state_len+self.control_len*self.T] + (1-gate)*grad_update

            return grad_update


def mpc_cost(x_t, u, verbose=False):
    cost = 0.0
    #x_t = x
    states = []
    states.append(x_t)
    #print(states)
    #print(u[0].shape, R.shape)
    for i in range(u.shape[0]):
        cost += x_t@Q@x_t.T + u[i:i+1]@R@u[i:i+1]
        x_t = (A@x_t.T + B@u[i:i+1]).T
        states.append(x_t)
    cost += x_t@Q@x_t.T
    if(verbose):
        print('*****************')
        print(states)
        print(u)
    return cost/3

def main(
    epoch_len=128,
    batch_size=32,
    T=10,
    iterations=4
    ):
    dataset = StateDataset(epoch_len=epoch_len, batch_size=batch_size)
    training_generator = torch.utils.data.DataLoader(dataset)

    model = MLP(1024,T=T).to(device).double()
    meta_opt = optim.Adam(model.parameters(), lr=1e-4)

    epochs = 0
    for X in training_generator:
        meta_train_loss = 0.0
        X = X.to(device).double()

        for j in range(batch_size):
            meta_opt.zero_grad()
            x = X[0,j:j+1]
            u = torch.tensor(np.zeros((T,1))).to(device).double().requires_grad_()

            mod = MLP2(512, T=T).to(device).double()
            reg_opt = optim.Adam(mod.parameters(), lr=1e-3)
            for k in range(40):
                reg_opt.zero_grad()
                u_pred = mod(x)
                if(k==29):
                    cost = mpc_cost(x, u_pred.T, verbose=True)
                    print('reg_cost 50:',cost.item())
                    u = u_pred.T
                elif(k==39):
                    cost = mpc_cost(x, u_pred.T, verbose=True)
                    print('reg_cost 60:',cost.item())
                else:
                    cost = mpc_cost(x, u_pred.T, verbose=False)
                cost.backward()
                reg_opt.step()
            #exit(0)
            u = u.detach().requires_grad_()
            meta_cost = 0.0
            grads_prev = None
            for k in range(iterations):
                cost = mpc_cost(x, u)
                grads = torch.autograd.grad(cost, u)[0].T
                if(k==0):
                    grads_prev = grads.detach()
                print(grads)
                full_input = torch.cat((x, u.T, grads, grads_prev), axis=1)
                grads_prev = grads.detach()
                u_pred = model(full_input).T
                cost = mpc_cost(x, u_pred, verbose=True)
                print('opt_cost:', cost.item())
                #meta_train_loss += cost.item()
                meta_cost += cost/iterations
                u = u_pred.detach()
                u.requires_grad = True
                #print(cost)
            
            meta_train_loss += meta_cost
            meta_cost.backward()
            meta_opt.step()
            #print('***************')

        print('epochs: ', epochs, meta_train_loss)

if __name__=='__main__':
    main()