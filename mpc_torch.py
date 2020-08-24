import numpy as np
from FTOCP import FTOCP
from LMPC import LMPC
import pdb
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import copy
import pickle
import random

import torch
from torch import nn, optim

import learn2learn as l2l

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
        #x_1 = A@x_list[i].T + B@u_list[i].T
        #loss += x_1.T@Q@x_1 + u_list[i]@R@u_list[i].T
        loss += (x_list[i]@Q@x_list[i].T + u_list[i]@R@u_list[i].T)#**0.5
    #print(loss)
    return loss

def main():
    device = torch.device('cuda')
    model = MLP(2, 256)
    model.to(device)
    #maml  = l2l.algorithms.MAML(model, lr=0.0005, first_order=False)
    opt = optim.Adam(model.parameters(), 0.003)
    checkpt = torch.load('maml_weights/new_maml')
    #model.load_state_dict(checkpt['model'])
    opt.load_state_dict(checkpt['opt_meta'])

    # Define system dynamics and cost
    A = np.array([[1,1],[0,1]])
    B = np.array([[0],[1]])
    Q = np.diag([1.0, 1.0]) #np.eye(2)
    R = 0.1#np.array([[1]])

    A_ten = torch.tensor(np.array([[1.0,1.0],[0.0,1.0]])).to(device).float()
    B_ten = torch.tensor(np.array([[0.0],[1.0]])).to(device).float()
    Q_ten = torch.tensor(0.01*np.diag([1.0, 1.0])).to(device).float()
    R_ten = torch.tensor([[0.1]]).to(device).float()

    print("Computing first feasible trajectory")
    
    # Initial Condition
    x0 = [-8.5, 1];

    N_feas = 10

    # ====================================================================================
    # Run simulation to compute feasible solution
    # ====================================================================================
    xcl_feasible = [x0]
    ucl_feasible =[]
    xt           = x0
    time         = 0
    costs = 0
    # time Loop (Perform the task until close to the origin)
    #while np.dot(xt, xt) > 10**(-15):
    for m in range(30):
        #learner = maml.clone()
        #model.load_state_dict(checkpt['model'])
        opt = optim.Adam(model.parameters(), lr=1e-3)
        #opt.load_state_dict(checkpt['optimizer'])

        xt = xcl_feasible[time] # Read measurements

        x = torch.tensor([xt]).to(device).float()

        for k in range(20):
            opt.zero_grad()           
            T = 10
            x_list = []
            x_list.append(x)
            u_list = []
            #print(x.shape)
            for i in range(T):
                u = model(x_list[i])
                #u = learner(x_list[i])
                x_next = A_ten@x_list[i].T + B_ten@u.T
                #print(x_next.shape)
                x_list.append(x_next.T)
                u_list.append(u)

            opt_loss = mpc_loss(x_list, u_list, A_ten, B_ten, Q_ten, R_ten)
            f_cost = opt_loss.item()
            #print(opt_loss.item())
            opt_loss.backward()
            opt.step()
        #print('************')
            #learner.adapt(opt_loss)
        #u = learner(x)
        costs += f_cost
        u = model(x)
        # Read input and apply it to the system
        ut = u.item()
        ucl_feasible.append(ut)
        xcl_feasible.append(ftocp_for_mpc.model(xcl_feasible[time], ut))
        time += 1

    print(np.round(np.array(xcl_feasible).T, decimals=2))
    print(np.round(np.array(ucl_feasible).T, decimals=2))
    print(costs)

if __name__== "__main__":
  main()
