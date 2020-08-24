import random
import numpy as np

import torch
from torch import nn, optim

import learn2learn as l2l

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
        #x_1 = A@x_list[i].T + B@u_list[i].T
        #loss += x_1.T@Q@x_1 + u_list[i]@R@u_list[i].T
        loss += (x_list[i+1]@Q@x_list[i+1].T + u_list[i]@R@u_list[i].T)#**0.5
    #print(loss)
    return loss

####WORK ON THIS FOR SOME REASON GRADIENTS DONT WORK ON OTHER ONE(PROBABLY BECAUSE OF LISTS)
def mpc_loss1(x, u, A, B, Q, R):
    x_1 = A@x.T + B@u.T
    loss = x_1.T@Q@x_1 + u@R@u.T
    #print(loss)
    return loss


def optimize(learner, x, A, B, Q, R):
    #learning
    T = 10
    x_list = []
    x_list.append(x)
    u_list = []
    #print(x.shape)
    for i in range(T):
        u = learner(x_list[i])
        x_next = A@x_list[i].T + B@u.T
        #print(x_next.shape)
        x_list.append(x_next.T)
        u_list.append(u)
    #print(x_list, u_list)
    #exit(0)
    opt_loss = mpc_loss(x_list, u_list, A, B, Q, R)
    #print('*****')
    #print(opt_loss)
    learner.adapt(opt_loss)

    #getting meta loss
    #u = learner(x)
    x_list = []
    x_list.append(x)
    u_list = []
    for i in range(T):
        u = learner(x_list[i])
        x_next = A@x_list[i].T + B@u.T
        x_list.append(x_next.T)
        u_list.append(u)
    #print('u: ', u)
    meta_loss = mpc_loss(x_list, u_list, A, B, Q, R)
    #print(meta_loss)

    return meta_loss

def main(
        batch_size=1,
        opt_steps=5,
        meta_lr=0.003,
        fast_lr=0.0005,
        meta_batch_size=128,
        adaptation_steps=1,
        epoch_len=128,
        epochs=1,
        cuda=True,
        seed=53):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device('cpu')
    if cuda and torch.cuda.device_count():
        torch.cuda.manual_seed(seed)
        device = torch.device('cuda')

    dataset = StateDataset(epoch_len=epoch_len, batch_size=meta_batch_size)
    training_generator = torch.utils.data.DataLoader(dataset)
    
    model = MLP(2, 256)
    #print(model.state_dict())
    model.to(device)   
    maml = l2l.algorithms.MAML(model, lr=fast_lr, first_order=False)
    #print(maml.state_dict())
    opt = optim.Adam(maml.parameters(), meta_lr)

    A = torch.tensor(np.array([[1.0,1.0],[0.0,1.0]])).to(device).float()
    B = torch.tensor(np.array([[0.0],[1.0]])).to(device).float()
    Q = torch.tensor(0.01*np.diag([1.0, 1.0])).to(device).float()
    R = torch.tensor([[0.1]]).to(device).float()
    loss = nn.CrossEntropyLoss(reduction='mean')
    
    for i in range(epochs):
        for X in training_generator:
            meta_train_loss = 0
            opt.zero_grad()
            X = X.to(device).float()
            
            meta_train_error = 0.0
            meta_train_accuracy = 0.0

            for j in range(meta_batch_size):
                learner = maml.clone()
                x = X[0,j:j+1]

                meta_loss = optimize(learner, x, A, B, Q, R)
                output = learner(x)
                #print(output)
                #meta_loss = mpc_loss1(x,output, A, B, Q, R)#loss(output, torch.tensor([1]).to(device).long())
                meta_loss.backward()
                meta_train_loss += meta_loss.item()

            print('meta_train_loss: ', meta_train_loss)
            for p in maml.parameters():
                #print(p.grad.data)
                p.grad.data.mul_(1.0 / meta_batch_size)
            opt.step()
    
    #maml = l2l.algorithms.MAML(model, lr=fast_lr, first_order=False)
    #opt = optim.Adam(maml.parameters(), meta_lr)
    #checkpt = torch.load('maml_weights/checkpt')
    #maml.load_state_dict(checkpt['model'])
    #opt.load_state_dict(checkpt['optimizer'])
    '''
    print("Checking opt:")
    for X in training_generator:
        X = X.to(device).float()
        for j in range(meta_batch_size):
            #learner = maml.clone()
            #x = X[0,j:j+1]
            x = torch.tensor([[13.91,  1.5]]).to(device).float()
            for k in range(30):
                opt.zero_grad()
                T = 15
                x_list = []
                x_list.append(x)
                u_list = []
                #print(x.shape)
                for i in range(T):
                    u = maml(x_list[i])
                    #u = learner(x_list[i])
                    x_next = A@x_list[i].T + B@u.T
                    #print(x_next.shape)
                    x_list.append(x_next.T)
                    u_list.append(u)
                #exit(0)
                opt_loss = mpc_loss(x_list, u_list, A, B, Q, R)
                print('*****')
                print(opt_loss)
                opt_loss.backward()
                opt.step()
                #learner.adapt(opt_loss)
                print('X',x_list)
                print('U',u_list)
            break
        break
        '''
    torch.save({'model' : maml.state_dict(), 'optimizer' : opt.state_dict()}, 'maml_weights/checkpt2')
    exit(0)
    '''
    re_maml  = l2l.algorithms.MAML(model, lr=fast_lr, first_order=False)
    re_maml.load_state_dict(torch.load('maml_weights/checkpt'))

    print("Checking remaml:")
    for X in training_generator:
        X = X.to(device).float()
        for j in range(meta_batch_size):
            learner = re_maml.clone()
            #x = X[0,j:j+1]
            x = torch.tensor([[-8.5,  1]]).to(device).float()
            for k in range(30):
                T = 2
                x_list = []
                x_list.append(x)
                u_list = []
                #print(x.shape)
                for i in range(T):
                    u = learner(x_list[i])
                    x_next = A@x_list[i].T + B@u.T
                    #print(x_next.shape)
                    x_list.append(x_next.T)
                    u_list.append(u)
                #exit(0)
                opt_loss = mpc_loss(x_list, u_list, A, B, Q, R)
                print('*****')
                print(opt_loss)
                learner.adapt(opt_loss)
                print('X',x_list)
                print('U',u_list)
            break
        break
    '''

if __name__=='__main__':
    main()