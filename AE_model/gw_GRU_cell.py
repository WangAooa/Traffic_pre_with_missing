import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys

import numpy as np

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:5") if torch.cuda.is_available() else torch.device("cpu")

class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('bnf,nm->bmf',(x,A))
        return x.contiguous()

class gcn(nn.Module):
    def __init__(self,c_in,c_out,num_node,dropout=0.3,support_len=3,order=2):
        super(gcn,self).__init__()
        self.nconv = nconv()
        c_in = (order*support_len+1)*c_in
        self.mlp = nn.Linear(c_in,c_out)
        self.num_node = num_node
        self.dropout = dropout
        self.order = order

        self.nodevec1 = nn.Parameter(torch.randn(num_node, 10).to(device), requires_grad=True).to(device)
        self.nodevec2 = nn.Parameter(torch.randn(10, num_node).to(device), requires_grad=True).to(device)

    def forward(self,x,support):
        adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
        support = support + [adp]

        out = [x]
        for a in support:
            x1 = self.nconv(x,a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1,a)
                out.append(x2)
                x1 = x2
        #print('in gcn, size of out: {}'.format(np.array(out).shape))
        h = torch.cat(out,dim=2)
        #print('in gcn, shape of h: {}'.format(h.shape))
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h

class gw_GUR_cell(torch.nn.Module):
    #GCG embeded in GRU
    def __init__(self, c_in,c_out, adj_mx, num_nodes, nonlinearity='tanh'):
        super(gw_GUR_cell, self).__init__()
        self._activation = torch.tanh if nonlinearity == 'tanh' else torch.relu
        # support other nonlinearities up here?
        self._num_nodes = num_nodes
        self.c_in = c_in
        self.c_out = c_out
        self.adj = adj_mx
        self.gcn_1 = gcn(c_in,2*c_out,num_nodes,support_len=len(adj_mx)+1)
        self.gcn_2 = gcn(c_in,c_out,num_nodes,support_len=len(adj_mx)+1)

    def forward(self, input, state):
        x = torch.cat([input,state],dim=2)
        value = torch.sigmoid(self.gcn_1(x,self.adj))
        r, u = torch.split(tensor=value, split_size_or_sections=self.c_out, dim=-1)
        # r = torch.reshape(r, (-1, self._num_nodes * self.c_out))
        # u = torch.reshape(u, (-1, self._num_nodes * self.c_out))

        x = torch.cat([input, r * state], dim=2)
        c = self.gcn_2(x, self.adj)
        if self._activation is not None:
            c = self._activation(c)
        #c:[B,N,out_dim]
        new_state = u * state + (1.0 - u) * c
        #new_state:[B,N,out_dim]
        return new_state

class gwnet_LSTM_cell(torch.nn.Module):
    #GCN + LSTM
    def __init__(self, c_in,c_out, adj_mx, num_nodes, nonlinearity='tanh'):
        super(gwnet_LSTM_cell, self).__init__()
        self._activation = torch.tanh if nonlinearity == 'tanh' else torch.relu
        self._num_nodes = num_nodes
        self.c_in = c_in
        self.c_out = c_out
        self.adj = adj_mx

        self.gcn = gcn(c_in, c_in * 2, num_nodes, support_len=len(adj_mx) + 1)
        self.lstm = nn.GRUCell(self.c_in * 2, self.c_out)

    def forward(self, input, state):
        # input : [B, N, F]
        batch,_,_ = input.shape

        gcn_result = self._activation(self.gcn(input, self.adj))

        gcn_result = torch.reshape(gcn_result,(-1,self.c_in * 2))
        state = torch.reshape(state, (-1, self.c_out))

        lstm_result = self.lstm(gcn_result,state)

        lstm_result = torch.reshape(lstm_result, (batch, self._num_nodes, self.c_out))

        return lstm_result

if __name__ == '__main__':
    adj = np.load('../PEMS(M)/dataset/W_228.npy')
    adj = torch.from_numpy(adj).to(device).to(torch.float)
    print(adj.dtype)

    model = gwnet_LSTM_cell(1, 32,[adj], 228).to(device)
    print(model)
    x = torch.zeros((32,228,1)).to(device)
    x_mask = torch.zeros((32,228,1)).to(device)
    time_lag = torch.zeros((32,228,1)).to(device)
    state = torch.zeros((32,228,32)).to(device)
    result = model(x,state)
    print(result.size())