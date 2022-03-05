import numpy as np
import torch
import torch.nn as nn

from AE_model.gw_GRU_cell import *
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:2") if torch.cuda.is_available() else torch.device("cpu")

###used the unit model : GCN + LSTM (2 layer)
class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('btnf,nm->btmf',(x,A))
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

        T,B,N,_ = x.shape

        out = [x]
        for a in support:
            x1 = self.nconv(x,a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1,a)
                out.append(x2)
                x1 = x2
        #print('in gcn, size of out: {}'.format(np.array(out).shape))
        h = torch.cat(out,dim=3)
        #print('in gcn, shape of h: {}'.format(h.shape))
        h = torch.reshape(h,(T*B,N,-1))
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = torch.reshape(h,(T,B,N,-1))
        return h

class LSTM_gwnet_cell(nn.Module):
    def __init__(self,rnn_unit, output_dim, adj, num_node, nonlinearity='tanh'):
        super(LSTM_gwnet_cell, self).__init__()
        self._activation = torch.tanh if nonlinearity == 'tanh' else torch.relu
        self.output_dim = output_dim
        self.adj = adj
        self.num_node = num_node
        self.rnn_unit = rnn_unit

        self.gru_layer = nn.GRUCell(self.rnn_unit, self.rnn_unit * 2)
        self.gcn_layer = gcn(self.rnn_unit * 2, self.output_dim,self.num_node,support_len=len(adj) + 1)

    def forward(self, inputs, state,mask = None, label = None):
        # inputs shape [T,B*N,F]
        #state shape [B*N, F]
        T,B_time_node,_ = inputs.shape

        hidden_states = []
        for t in range(T):
            output = inputs[t]

            next_hidden_state = self._activation(self.gru_layer(output, state))

            hidden_states.append(next_hidden_state)

            #normalization
            # output = next_hidden_state + output
            # output = torch.reshape(output, (-1,self.rnn_unit))
            # output = self.normalization(output)
            # output = torch.reshape(output, (batch_size,self.num_node,self.rnn_unit))

        hidden_states = torch.stack(hidden_states)

        gcn_input = torch.reshape(hidden_states,(T, (int)(B_time_node / self.num_node), self.num_node, -1))

        gcn_result = self._activation(self.gcn_layer(gcn_input, self.adj))
        gcn_result = torch.reshape(gcn_result,(T,B_time_node,-1))

        return gcn_result,hidden_states[-1]


class EncoderModel(nn.Module):
    def __init__(self, intput_dim, adj, rnn_unit, num_node, num_rnn_layers):
        super(EncoderModel, self).__init__()
        self.input_dim = intput_dim
        self.adj = adj
        self.rnn_unit = rnn_unit
        self.num_node = num_node
        self.num_rnn_layers = num_rnn_layers

        self.input_projection = nn.Linear(in_features=self.input_dim * 2, out_features=256)
        self.input_projection_2 = nn.Linear(256,self.rnn_unit)

        self.gwgru = nn.ModuleList([LSTM_gwnet_cell(self.rnn_unit, self.rnn_unit, self.adj, self.num_node),
                                    LSTM_gwnet_cell(self.rnn_unit, self.rnn_unit, self.adj, self.num_node)])

        self.gwgru_reverse = nn.ModuleList([LSTM_gwnet_cell(self.rnn_unit, self.rnn_unit, self.adj, self.num_node),
                                            LSTM_gwnet_cell(self.rnn_unit, self.rnn_unit, self.adj, self.num_node)])
        #
        self.linear_dense_layer = nn.Linear(4*self.rnn_unit, 256)
        self.linear_dense_layer_2 = nn.Linear(256, self.rnn_unit * 2)
        # self.stgru = nn.ModuleList([STGRU_cell(self.input_dim + self.rnn_unit, self.rnn_unit, self.adj, self.num_node)]
        #                             )

    def forward(self, inputs, mask):
        #inputs shape [T,B,N,F]

        distribution = torch.empty(inputs.shape).normal_(mean=0,std=0.01).to(device)
        inputs = inputs + distribution

        time_step,batch_size, _,_ = inputs.size()

        inputs = torch.cat([inputs,mask], dim=3)

        hidden_state = torch.zeros((self.num_rnn_layers, batch_size*self.num_node, self.rnn_unit*2),device=device)
        hidden_state_reverse = torch.zeros((self.num_rnn_layers, batch_size*self.num_node, self.rnn_unit*2),device=device)

        inputs = torch.reshape(inputs, (time_step, batch_size*self.num_node,-1))

        inputs_projection = F.relu(self.input_projection(inputs))
        inputs_projection = F.relu(self.input_projection_2(inputs_projection))
        #inputs_projection shape [T, B*N, F]

        hidden_states = []
        output = inputs_projection
        for layer_num, gwgru_layer in enumerate(self.gwgru):
            output, next_hidden_state = gwgru_layer(output, hidden_state[layer_num])

            hidden_states.append(next_hidden_state)

        hidden_state = torch.stack(hidden_states)

        inputs_reverse = torch.flip(inputs, [0])
        inputs_projection = F.relu(self.input_projection(inputs_reverse))
        inputs_projection = F.relu(self.input_projection_2(inputs_projection))

        hidden_states_reverses = []
        output = inputs_projection
        for layer_num, gwgru_layer in enumerate(self.gwgru_reverse):
            output, next_hidden_state = gwgru_layer(output, hidden_state_reverse[layer_num])

            hidden_states_reverses.append(next_hidden_state)

        hidden_state_reverse = torch.stack(hidden_states_reverses)

        hidden_state = torch.cat([hidden_state, hidden_state_reverse], dim=2)
        hidden_state = F.relu(self.linear_dense_layer(hidden_state))
        hidden_state = self.linear_dense_layer_2(hidden_state)

        return hidden_state

class DecoderModel(nn.Module):
    def __init__(self, intput_dim, adj, rnn_unit, num_node, num_rnn_layers, output_dim, horrizon,cl_decay_steps=1000):
        super(DecoderModel, self).__init__()
        self.input_dim = intput_dim
        self.adj = adj
        self.rnn_unit = rnn_unit
        self.num_node = num_node
        self.num_rnn_layers = num_rnn_layers
        self.output_dim = output_dim
        self.horrizon = horrizon
        self.cl_decay_steps = cl_decay_steps
        # self.stgru_layers = nn.ModuleList([gw_GUR_cell(2*self.rnn_unit, self.rnn_unit, self.adj,self.num_node),
        #                                    gw_GUR_cell(2*self.rnn_unit, self.rnn_unit, self.adj,self.num_node)])

        self.input_projection = nn.Linear(in_features=self.output_dim, out_features=256)
        self.input_projection_2 = nn.Linear(256, self.rnn_unit)

        self.stgru_layers = nn.ModuleList(
            [LSTM_gwnet_cell(self.rnn_unit, self.rnn_unit, self.adj, self.num_node),
             LSTM_gwnet_cell(self.rnn_unit, self.rnn_unit, self.adj, self.num_node)])

        # self.stgru_layers = nn.ModuleList(
        #     [STGRU_cell(self.output_dim + self.rnn_unit, self.rnn_unit, self.adj, self.num_node)])

        self.projection_layer = nn.Linear(self.rnn_unit, 256)
        self.projection_layer_2 = nn.Linear(256, self.output_dim)

    def _compute_sampling_threshold(self, batches_seen):
        return self.cl_decay_steps / (
                self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps))

    def forward(self, encoder_hidden_state, mask = None,labels=None, batches_seen=None):
        batch_size = (int)(encoder_hidden_state.size(1) / self.num_node)

        go_symbol = torch.zeros((self.horrizon,batch_size*self.num_node, self.output_dim),
                                device=device)
        decoder_hidden_state = encoder_hidden_state
        decoder_input = go_symbol

        distribution = torch.empty(decoder_input.shape).normal_(mean=0, std=0.01).to(device)
        output = decoder_input + distribution
        # output dim [T, B*N, F]

        output = F.relu(self.input_projection(output))
        output = F.relu(self.input_projection_2(output))

        hidden_states = []
        for layer_num, stgru_layer in enumerate(self.stgru_layers):
            output, next_hidden_state = stgru_layer(output, decoder_hidden_state[layer_num])
            hidden_states.append(next_hidden_state)

        decoder_output = F.relu(self.projection_layer(output))
        decoder_output = self.projection_layer_2(decoder_output)

        return decoder_output


class Model(nn.Module):
    def __init__(self, intput_dim, adj, rnn_unit, num_node, num_rnn_layers, output_dim, horrizon,seq_len,cl_decay_steps= 1000):
        super(Model, self).__init__()
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


    def forward(self, inputs, mask,labels=None, batches_seen=None):
        """
        seq2seq forward pass
        :param labels: shape (horizon, batch_size, num_sensor * output)
        :param batches_seen: batches seen till now
        :return: output: (self.horizon, batch_size, self.num_nodes * self.output_dim)
        """
        B,T,N,F = inputs.shape

        inputs = inputs.permute(1,0,2,3)
        mask = mask.permute(1,0,2,3)

        if labels is not None:
            labels = labels.permute(1,0,2,3)
        encoder_hidden_state = self.encoder_model(inputs, mask)

        outputs = self.decoder_model(encoder_hidden_state,mask, labels, batches_seen=batches_seen)
        # output:[horrizon, B, N, output_dim]
        outputs = torch.reshape(outputs, (self.horrizon,B,N,self.output_dim)).permute(1,0,2,3)
        return outputs

if __name__ == '__main__':
    adj = np.load('../PEMS(M)/dataset/W_228.npy')
    adj = torch.from_numpy(adj).to(device).to(torch.float)
    print(adj.dtype)

    model = Model(1, [adj], 32, 228, 2, 1, 12, 12).to(device)
    print(model)
    x = torch.zeros((32,12,228,1)).to(device)
    x_mask = torch.zeros((32,12,228,1)).to(device)
    time_lag = torch.zeros((32,12,228,1)).to(device)

    result = model(x,x_mask)
    print(result.size())
    # for name, para in model.named_parameters():
    #     if 'encoder' in name:
    #         para.requires_grad = False
    #
    # for name, para in model.named_parameters():
    #     if (para.requires_grad):
    #         print(name)
    #         print(para)
    #         print(para.requires_grad)
    #         print('________________')