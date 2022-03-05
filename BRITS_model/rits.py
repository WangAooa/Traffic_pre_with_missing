import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.autograd import Variable
from torch.nn.parameter import Parameter

import math
import numpy as np

device = torch.device("cuda:5") if torch.cuda.is_available() else torch.device("cpu")
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class TemporalDecay(nn.Module):
    def __init__(self, input_size, rnn_hid_size):
        super(TemporalDecay, self).__init__()
        self.rnn_hid_size = rnn_hid_size
        self.build(input_size)

    def build(self, input_size):
        self.W = Parameter(torch.Tensor(self.rnn_hid_size, input_size))
        self.b = Parameter(torch.Tensor(self.rnn_hid_size))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(0))
        self.W.data.uniform_(-stdv, stdv)
        if self.b is not None:
            self.b.data.uniform_(-stdv, stdv)

    def forward(self, d):
        gamma = F.relu(F.linear(d, self.W, self.b))
        gamma = torch.exp(-gamma)
        return gamma


class rits(nn.Module):
    def __init__(self, input_dim, rnn_dim, batch, num_node):
        super(rits, self).__init__()
        self.input_dim = input_dim
        self.rnn_dim = rnn_dim
        self.batch = batch
        self.num_node = num_node


        self.input_layer = nn.Linear(self.input_dim * 2, self.rnn_dim * 2)
        self.rnn = nn.LSTMCell(self.rnn_dim * 2, self.rnn_dim)
        self.temp_decay = TemporalDecay(input_size=1, rnn_hid_size=self.rnn_dim)
        self.out_layer = nn.Linear(self.rnn_dim, 1)

        #self.gat = GAT(self.rnn_dim,64,1,0.5,0.2,8)


    def forward(self, x,x_mask, time_lag, h, c):

        gamma = self.temp_decay(time_lag)  # [B*N, rnn_dim]
        h = h * gamma
        x_h = self.out_layer(h)

        x_c = x_mask * x + (1 - x_mask) * x_h

        inputs = torch.cat([x_c, x_mask], dim=1)
        inputs = self.input_layer(inputs)

        h, c = self.rnn(inputs, (h, c))
        return x_c, (h,c)

class Model_rits(nn.Module):
    def __init__(self, input_dim, rnn_dim, batch, num_node, time_step, pre_step):
        super(Model_rits, self).__init__()
        self.input_dim = input_dim
        self.rnn_dim = rnn_dim
        self.batch = batch
        self.num_node = num_node
        self.time_step = time_step
        self.pre_step = pre_step

        self.rnn = rits(input_dim, rnn_dim, batch, num_node)

    def forward(self, x, x_mask, time_lag):
        #x[B,T,N,F] -> [T,B,N,F]
        x = x.permute(1, 0, 2, 3)
        x_mask = x_mask.permute(1, 0, 2, 3)
        time_lag = time_lag.permute(1, 0, 2, 3)
        x = torch.reshape(x, (self.time_step, self.batch * self.num_node, 1))
        x_mask = torch.reshape(x_mask, (self.time_step, self.batch * self.num_node, 1))
        time_lag = torch.reshape(time_lag, (self.time_step, self.batch * self.num_node, 1))

        imputations, prediction = [], []

        h = torch.zeros((self.batch * self.num_node, self.rnn_dim)).to(device)
        c = torch.zeros((self.batch * self.num_node, self.rnn_dim)).to(device)

        for t in range(x.size()[0]):
            x_temp = x[t]
            mask_temp = x_mask[t]
            d = time_lag[t]
            x_c, (h,c) = self.rnn(x_temp, mask_temp, d, h, c)
            # x_c = self.rnn(x[t], x_mask[t], time_lag[t])
            imputations.append(x_c)


        imputations = torch.stack(imputations)

        for i in range(self.pre_step):
            x_temp = torch.ones((self.batch * self.num_node, 1)).to(device)
            mask_temp = torch.zeros((self.batch * self.num_node, 1)).to(device)
            d = torch.ones((self.batch * self.num_node, 1)).to(device)
            x_c, (h,c) = self.rnn(x_temp, mask_temp, d, h, c)
            prediction.append(x_c)

        prediction = torch.stack(prediction)
        return imputations, prediction
        # return imputations
#模型维度 [B,T,N,F]
#经过FNN变成 [B,T,N,F']升纬度
#经过rits 循环网络进行填补
#同时进行图网络进行填补
#对两种填补进行整合

class Model(nn.Module):
    def __init__(self, input_dim, rnn_dim, batch, num_node, time_step, pre_step, adj):
        super(Model, self).__init__()
        self.input_dim = input_dim
        self.rnn_dim = rnn_dim
        self.batch = batch
        self.num_node = num_node
        self.time_step = time_step
        self.pre_step = pre_step
        self.adj = adj

        self.input_layer = nn.Linear(self.input_dim * 2, self.rnn_dim * 2)
        self.rnn = nn.LSTMCell(self.rnn_dim * 2, self.rnn_dim)
        self.temp_decay = TemporalDecay(input_size = 1, rnn_hid_size = self.rnn_dim)

        self.out_layer = nn.Linear(self.rnn_dim, 1)

    def forward(self, x,x_mask, time_lag, y_missing):
        batch = x.size()[0]
        time_step = x.size()[1]
        num_node = x.size()[2]

        x = x.permute(1,0,2,3)
        x_mask = x_mask.permute(1,0,2,3)
        time_lag = time_lag.permute(1,0,2,3)
        x = torch.reshape(x, (time_step, batch * num_node, 1))
        x_mask = torch.reshape(x_mask, (time_step, batch * num_node, 1))
        time_lag = torch.reshape(time_lag, (time_step, batch * num_node, 1))

        y_missing = y_missing.permute(1,0,2,3)
        y_missing = torch.reshape(y_missing, (time_step, batch * num_node, 1))


        h = Variable(torch.zeros((batch * num_node, self.rnn_dim)))
        c = Variable(torch.zeros((batch * num_node, self.rnn_dim)))

        # if torch.cuda.is_available():
        #     h, c = h.cuda(), c.cuda()

        impute_loss = 0.0

        imputations, prediction = [], []

        for t in range(x.size()[0]):
            x_temp = x[t]
            mask_temp = x_mask[t]
            d = time_lag[t]

            y_missing_temp = y_missing[t]

            gamma = self.temp_decay(d)   #[B*N, rnn_dim]
            h = h * gamma
            x_h = self.out_layer(h)

            impute_loss += torch.sum(torch.abs(x_temp - x_h) * mask_temp) / (torch.sum(mask_temp) + 1e-5)

            x_c = mask_temp * x_temp  + (1 - mask_temp) * x_h

            inputs = torch.cat([x_c, mask_temp], dim=1)
            inputs = self.input_layer(inputs)

            h, c = self.rnn(inputs, (h, c))

            #imputations.append(x_c.unsqueeze(dim=1))
            imputations.append(x_c)

        #imputations = torch.cat(imputations, dim=1)
        imputations = torch.stack(imputations)

        for i in range(self.pre_step):
            x_temp = torch.ones((self.batch * self.num_node, 1)).to(device)
            mask_temp = torch.zeros((self.batch * self.num_node, 1)).to(device)
            d = torch.ones((self.batch * self.num_node, 1)).to(device)
            x_c, (h,c) = self.rnn(x_temp, mask_temp, d, h, c)
            prediction.append(x_c)

        prediction = torch.stack(prediction)

        return impute_loss / x.size()[0],  imputations

if __name__ == '__main__':
    adj = np.load('../PEMS(M)/dataset/W_228.npy')
    adj = torch.from_numpy(adj).to(device).float()


    model = Model_1(1,32, 32, 228, 12, 12, adj).to(device)
    print(model)
    x = torch.zeros((32,12,228,1)).to(device).float()
    x_mask = torch.zeros((32,12,228,1)).to(device).float()
    time_lag = torch.zeros((32,12,228,1)).to(device).float()

    result, pre = model(x,x_mask,time_lag)

    print(x_mask.size())
    print(result.size())
    print(pre.size())

