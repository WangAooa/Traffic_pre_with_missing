import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np


device = torch.device("cuda:4") if torch.cuda.is_available() else torch.device("cpu")
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class spatio_conv_layer(nn.Module):
    def __init__(self, c_in,c_out, Lk):
        super(spatio_conv_layer, self).__init__()
        self.Lk = Lk
        self.c_in = c_in
        self.c_out = c_out
        self.theta = nn.Parameter(torch.FloatTensor(c_in, c_out))
        self.b = nn.Parameter(torch.FloatTensor(1, 1,c_out))

        self.theta1 = nn.Parameter(torch.FloatTensor(c_out,c_out))
        self.b1 = nn.Parameter(torch.FloatTensor(1,1,c_out))
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.theta, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.theta)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.b, -bound, bound)

        init.kaiming_uniform_(self.theta1, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.theta1)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.b1, -bound, bound)

    def forward(self, input, state):
        #input:[B,N,input_dim] state:[B,N,outpu_dim]
        batch_size = input.shape[0]
        num_node = input.shape[1]
        # inputs = torch.reshape(inputs, (batch_size, self._num_nodes, -1))
        # state = torch.reshape(state, (batch_size, self._num_nodes, -1))
        inputs_and_state = torch.cat([input, state], dim=2)

        lfs = torch.einsum("ij,jkl->kil", [self.Lk, inputs_and_state.permute(1,0,2)])
        #lfs:[B,N,input_dim]

        t2 = F.relu(torch.matmul(lfs, self.theta)) + self.b
        return t2

class STGRU_cell(torch.nn.Module):
    def __init__(self, c_in,c_out, adj_mx, num_nodes, nonlinearity='tanh'):
        super().__init__()
        self._activation = torch.tanh if nonlinearity == 'tanh' else torch.relu
        # support other nonlinearities up here?
        self._num_nodes = num_nodes
        self.c_in = c_in
        self.c_out = c_out
        self.adj = adj_mx
        self.gcn_1 = spatio_conv_layer(c_in,2*c_out,adj_mx)
        self.gcn_2 = spatio_conv_layer(c_in,c_out,adj_mx)

    def forward(self, input, state):
        value = torch.sigmoid(self.gcn_1(input, state))
        r, u = torch.split(tensor=value, split_size_or_sections=self.c_out, dim=-1)
        # r = torch.reshape(r, (-1, self._num_nodes * self.c_out))
        # u = torch.reshape(u, (-1, self._num_nodes * self.c_out))

        c = self.gcn_2(input, r * state)
        if self._activation is not None:
            c = self._activation(c)
        #c:[B,N,out_dim]
        new_state = u * state + (1.0 - u) * c
        #new_state:[B,N,out_dim]
        return new_state

class EncoderModel(nn.Module):
    def __init__(self, intput_dim, adj, rnn_unit, num_node, num_rnn_layers):
        super(EncoderModel, self).__init__()
        self.input_dim = intput_dim
        self.adj = adj
        self.rnn_unit = rnn_unit
        self.num_node = num_node
        self.num_rnn_layers = num_rnn_layers
        self.stgru = nn.ModuleList([STGRU_cell(self.input_dim+self.rnn_unit, self.rnn_unit, self.adj,self.num_node),
                                    STGRU_cell(2*self.rnn_unit, self.rnn_unit, self.adj,self.num_node)])

        # self.stgru = nn.ModuleList([STGRU_cell(self.input_dim + self.rnn_unit, self.rnn_unit, self.adj, self.num_node)]
        #                             )

    def forward(self, inputs, hidden_state = None):
        batch_size, _,_ = inputs.size()
        if hidden_state is None:
            hidden_state = torch.zeros((self.num_rnn_layers, batch_size, self.num_node, self.rnn_unit),
                                       device=device)
        hidden_states = []
        output = inputs
        for layer_num, stgru_layer in enumerate(self.stgru):
            next_hidden_state = stgru_layer(output, hidden_state[layer_num])

            hidden_states.append(next_hidden_state)
            output = next_hidden_state

        return output, torch.stack(hidden_states)

class DecoderModel(nn.Module):
    def __init__(self, intput_dim, adj, rnn_unit, num_node, num_rnn_layers, output_dim, horrizon):
        super(DecoderModel, self).__init__()
        self.input_dim = intput_dim
        self.adj = adj
        self.rnn_unit = rnn_unit
        self.num_node = num_node
        self.num_rnn_layers = num_rnn_layers
        self.output_dim = output_dim
        self.horrizon = horrizon
        self.stgru_layers = nn.ModuleList([STGRU_cell(self.output_dim+self.rnn_unit, self.rnn_unit, self.adj,self.num_node),
                                           STGRU_cell(2*self.rnn_unit, self.rnn_unit, self.adj,self.num_node)])

        # self.stgru_layers = nn.ModuleList(
        #     [STGRU_cell(self.output_dim + self.rnn_unit, self.rnn_unit, self.adj, self.num_node)])

        self.projection_layer = nn.Linear(self.rnn_unit, self.output_dim)

    def forward(self, inputs, hidden_state=None):
        hidden_states = []
        output = inputs
        for layer_num, stgru_layer in enumerate(self.stgru_layers):
            next_hidden_state = stgru_layer(output, hidden_state[layer_num])
            hidden_states.append(next_hidden_state)
            output = next_hidden_state

        #projected = self.projection_layer(output.view(-1, self.rnn_unit))
        #output = projected.view(-1, self.num_node, self.output_dim)

        output = self.projection_layer(output)

        return output, torch.stack(hidden_states)

class STGRU(nn.Module):
    def __init__(self, adj,num_node, intput_dim=1,  rnn_unit=64, num_rnn_layers=2, output_dim=1, horrizon=12,seq_len=12,cl_decay_steps= 1000):
        super(STGRU, self).__init__()
        self.input_dim = intput_dim
        self.adj = adj
        self.rnn_unit = rnn_unit
        self.num_node = num_node
        self.num_rnn_layers = num_rnn_layers
        self.output_dim = output_dim
        self.horrizon = horrizon
        self.seq_len = seq_len
        self.encoder_model = EncoderModel(self.input_dim,self.adj,self.rnn_unit,self.num_node,self.num_rnn_layers)
        self.decoder_model = DecoderModel(self.input_dim,self.adj,self.rnn_unit,self.num_node,self.num_rnn_layers,self.output_dim,self.horrizon)
        self.cl_decay_steps = cl_decay_steps
    def _compute_sampling_threshold(self, batches_seen):
        return self.cl_decay_steps / (
                self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps))

    def encoder(self, inputs):
        """
        encoder forward pass on t time steps
        :param inputs: shape (seq_len, batch_size, num_sensor * input_dim)
        :return: encoder_hidden_state: (num_layers, batch_size, self.hidden_state_size)
        """
        encoder_hidden_state = None
        for t in range(self.seq_len):
            _, encoder_hidden_state = self.encoder_model(inputs[t], encoder_hidden_state)

        return encoder_hidden_state

    def decoder(self, encoder_hidden_state, labels=None, batches_seen=None):
        """
        Decoder forward pass
        :param encoder_hidden_state: (num_layers, batch_size, self.hidden_state_size)
        :param labels: (self.horizon, batch_size, self.num_nodes * self.output_dim) [optional, not exist for inference]
        :param batches_seen: global step [optional, not exist for inference]
        :return: output: (self.horizon, batch_size, self.num_nodes * self.output_dim)
        """
        batch_size = encoder_hidden_state.size(1)
        go_symbol = torch.zeros((batch_size, self.num_node, self.output_dim),
                                device=device)
        decoder_hidden_state = encoder_hidden_state
        decoder_input = go_symbol

        outputs = []

        for t in range(self.horrizon):
            decoder_output, decoder_hidden_state = self.decoder_model(decoder_input,
                                                                      decoder_hidden_state)
            decoder_input = decoder_output
            outputs.append(decoder_output)
            # if self.training :
            #     c = np.random.uniform(0, 1)
            #     if c < self._compute_sampling_threshold(batches_seen):
            #         decoder_input = labels[t]
        outputs = torch.stack(outputs)
        #output:[horrizon, B, N, output_dim]
        return outputs

    def forward(self, inputs, labels=None, batches_seen=None):
        """
        seq2seq forward pass
        :param inputs: shape (seq_len, batch_size, num_sensor * input_dim)
        :param labels: shape (horizon, batch_size, num_sensor * output)
        :param batches_seen: batches seen till now
        :return: output: (self.horizon, batch_size, self.num_nodes * self.output_dim)
        """
        inputs = inputs.permute(1,0,2,3)
        if labels is not None:
            labels = labels.permute(1,0,2,3)

        encoder_hidden_state = self.encoder(inputs)
        outputs = self.decoder(encoder_hidden_state, labels, batches_seen=batches_seen)
        # output:[horrizon, B, N, output_dim]
        outputs = outputs.permute(1,0,2,3)
        return outputs

if __name__ == '__main__':
    adj = np.load('../../PEMS(M)/dataset/W_228.npy')
    adj = torch.from_numpy(adj).to(device).float()


    model = STGRU(adj=adj, num_node=228).to(device)
    print(model)
    x = torch.zeros((32,12,228,1)).to(device).float()
    x_mask = torch.zeros((32,12,228,1)).to(device).float()
    time_lag = torch.zeros((32,1,228,12)).to(device).float()

    result = model(x,x_mask,10)

    print(x_mask.size())
    print(result.size())