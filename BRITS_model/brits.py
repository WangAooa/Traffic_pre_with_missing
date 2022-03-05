import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.autograd import Variable
from torch.nn.parameter import Parameter

import math
import numpy as np

from BRITS_model.rits import Model_rits
device = torch.device("cuda:5") if torch.cuda.is_available() else torch.device("cpu")
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class brits(nn.Module):
    def __init__(self, input_dim, rnn_dim, batch, num_node, time_step, pre_step):
        super(brits, self).__init__()
        self.input_dim = input_dim
        self.rnn_dim = rnn_dim
        self.batch = batch
        self.num_node = num_node
        self.time_step = time_step
        self.pre_step = pre_step

        self.rits_forward = Model_rits(input_dim, rnn_dim, batch, num_node, time_step, pre_step)
        self.rits_backward = Model_rits(input_dim, rnn_dim, batch, num_node, time_step, pre_step)

    def forward(self, x, x_mask, time_lag, time_lag_reverse):
        x_reverse = torch.flip(x, [1])
        x_mask_reverse = torch.flip(x_mask, [1])

        impute_forward, predict_forward = self.rits_forward(x, x_mask, time_lag)
        impute_forward = torch.reshape(impute_forward, (self.time_step, self.batch, self.num_node, self.input_dim)).permute(1,0,2,3)
        predict_forward = torch.reshape(predict_forward, (self.pre_step, self.batch, self.num_node, self.input_dim)).permute(1,0,2,3)

        impute_backward,_ = self.rits_backward(x_reverse, x_mask_reverse, time_lag_reverse)
        #
        impute_backward = torch.flip(impute_backward, [0])
        impute_backward = torch.reshape(impute_backward, (self.time_step, self.batch, self.num_node, self.input_dim)).permute(1,0,2,3)
        impute = (impute_forward + impute_backward) / 2

        return impute, predict_forward

if __name__ == '__main__':
    # adj = np.load('../PEMS(M)/dataset/W_228.npy')
    # adj = torch.from_numpy(adj).to(device).to(torch.float)
    # print(adj.dtype)
    #
    # model = brits(1,32, 32, 228, 12, 12).to(device)
    # print(model)
    # x = torch.zeros((32,12,228,1)).to(device)
    # x_mask = torch.zeros((32,12,228,1)).to(device)
    # time_lag = torch.zeros((32,12,228,1)).to(device)
    # time_lag_reverse = time_lag
    #
    # result, pre = model(x,x_mask,time_lag, time_lag_reverse)
    # print(result.size())
    # print(pre.size())
    a = np.arange(0,16)
    a = torch.from_numpy(a).reshape(2,2,2,2)
    print(a)
    a = torch.flip(a,[0])
    print(a)

