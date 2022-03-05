import numpy as np
import torch
import torch.nn as nn

from AE_model.gw_GRU_cell import *
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:5") if torch.cuda.is_available() else torch.device("cpu")

###used the unit model : GCN + LSTM (2 layer)

class EncoderModel(nn.Module):
    def __init__(self, intput_dim, adj, rnn_unit, num_node, num_rnn_layers):
        super(EncoderModel, self).__init__()
        self.input_dim = intput_dim
        self.adj = adj
        self.rnn_unit = rnn_unit
        self.num_node = num_node
        self.num_rnn_layers = num_rnn_layers
        # self.gwgru = nn.ModuleList([gw_GUR_cell(self.rnn_unit * 2, self.rnn_unit, self.adj,self.num_node),
        #                             gw_GUR_cell(2*self.rnn_unit, self.rnn_unit, self.adj,self.num_node)])
        #
        # self.gwgru_reverse = nn.ModuleList([gw_GUR_cell(self.rnn_unit * 2, self.rnn_unit, self.adj, self.num_node),
        #                                     gw_GUR_cell(self.rnn_unit * 2, self.rnn_unit, self.adj, self.num_node)])

        # self.input_projection = nn.Linear(in_features=self.input_dim * 2, out_features= self.rnn_unit)
        self.input_projection = nn.Linear(in_features=self.input_dim * 2, out_features=256)
        self.input_projection_2 = nn.Linear(256,self.rnn_unit)

        # self.gwgru = nn.ModuleList([gwnet_LSTM_cell(self.rnn_unit, self.rnn_unit, self.adj, self.num_node),
        #                             gwnet_LSTM_cell(self.rnn_unit, self.rnn_unit, self.adj, self.num_node)])
        #
        # self.gwgru_reverse = nn.ModuleList([gwnet_LSTM_cell(self.rnn_unit, self.rnn_unit, self.adj, self.num_node),
        #                                     gwnet_LSTM_cell(self.rnn_unit, self.rnn_unit, self.adj, self.num_node)])
        #
        self.gwgru = nn.ModuleList()
        self.gwgru_reverse = nn.ModuleList()
        # self.batch_norm = nn.ModuleList()
        # self.batch_norm_reverse = nn.ModuleList()

        self.linear_dense_layer = nn.Linear(2*self.rnn_unit, 256)
        self.linear_dense_layer_2 = nn.Linear(256, self.rnn_unit)
        # self.stgru = nn.ModuleList([STGRU_cell(self.input_dim + self.rnn_unit, self.rnn_unit, self.adj, self.num_node)]
        #                             )
        for i in range(num_rnn_layers):
            self.gwgru.append(gwnet_LSTM_cell(self.rnn_unit, self.rnn_unit, self.adj, self.num_node))
            self.gwgru_reverse.append(gwnet_LSTM_cell(self.rnn_unit, self.rnn_unit, self.adj, self.num_node))
            # self.batch_norm.append(nn.BatchNorm1d(self.rnn_unit))
            # self.batch_norm_reverse.append((nn.BatchNorm1d(self.rnn_unit)))
        # for i in range(num_rnn_layers - 2):
        #     self.batch_norm.append(nn.BatchNorm1d(self.rnn_unit))
        #     self.batch_norm_reverse.append((nn.BatchNorm1d(self.rnn_unit)))

    def forward(self, inputs, mask):
        #inputs shape [T,B,N,F]

        distribution = torch.empty(inputs.shape).normal_(mean=0,std=0.01).to(device)
        inputs = inputs + distribution

        time_step,batch_size, _,_ = inputs.size()

        inputs = torch.cat([inputs,mask], dim=3)

        hidden_state = torch.zeros((self.num_rnn_layers, batch_size, self.num_node, self.rnn_unit),device=device)
        hidden_state_reverse = torch.zeros((self.num_rnn_layers, batch_size, self.num_node, self.rnn_unit),device=device)

        for t in range(time_step):
            hidden_states = []
            output = inputs[t]

            # output = F.relu(self.input_projection(output))
            output = F.relu(self.input_projection(output))
            output = F.relu(self.input_projection_2(output))

            for layer_num, gwgru_layer in enumerate(self.gwgru):
                next_hidden_state = gwgru_layer(output, hidden_state[layer_num])

                hidden_states.append(next_hidden_state)
                # if self.num_rnn_layers > 2 and 1 <= layer_num < (self.num_rnn_layers - 1):
                #     output = output + next_hidden_state
                #     shape = output.shape
                #     output = torch.reshape(output, (-1, self.rnn_unit))
                #     output = self.batch_norm[layer_num - 1](output)
                #     output = torch.reshape(output, shape)
                # else:
                #     output = next_hidden_state
                output = output + next_hidden_state
                # shape = output.shape
                # output = torch.reshape(output, (-1, self.rnn_unit))
                #output = F.relu(self.batch_norm[layer_num](output))
                output = F.relu(output)
                # output = torch.reshape(output, shape)

            hidden_state = torch.stack(hidden_states)

        inputs_reverse = torch.flip(inputs, [0])

        for t in range(time_step):
            hidden_states = []
            output = inputs_reverse[t]

            output = F.relu(self.input_projection(output))
            output = F.relu(self.input_projection_2(output))

            for layer_num, gwgru_layer in enumerate(self.gwgru_reverse):
                next_hidden_state = gwgru_layer(output, hidden_state_reverse[layer_num])

                hidden_states.append(next_hidden_state)

                # if self.num_rnn_layers > 2 and 1 <= layer_num < (self.num_rnn_layers - 1):
                #     output = output + next_hidden_state
                #     shape = output.shape
                #     output = torch.reshape(output, (-1, self.rnn_unit))
                #     output = self.batch_norm_reverse[layer_num - 1](output)
                #     output = torch.reshape(output,shape)
                # else:
                #     output = next_hidden_state
                output = output + next_hidden_state
                # shape = output.shape
                # output = torch.reshape(output, (-1, self.rnn_unit))
                # output = F.relu(self.batch_norm[layer_num](output))
                output = F.relu(output)
                # output = torch.reshape(output, shape)

            hidden_state_reverse = torch.stack(hidden_states)

        hidden_state = torch.cat([hidden_state, hidden_state_reverse], dim=3)
        hidden_state = F.relu(self.linear_dense_layer(hidden_state))
        hidden_state = self.linear_dense_layer_2(hidden_state)

        # # dropout
        # hidden_state = F.dropout(hidden_state, 0.3, training=self.training)


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

        # self.stgru_layers = nn.ModuleList(
        #     [gwnet_LSTM_cell(self.rnn_unit, self.rnn_unit, self.adj, self.num_node),
        #      gwnet_LSTM_cell(self.rnn_unit, self.rnn_unit, self.adj, self.num_node)])

        self.gwgru = nn.ModuleList()
        #self.batch_norm = nn.ModuleList()

        for i in range(num_rnn_layers):
            self.gwgru.append(gwnet_LSTM_cell(self.rnn_unit, self.rnn_unit, self.adj, self.num_node))
            #self.batch_norm.append(nn.BatchNorm1d(self.rnn_unit))

        self.projection_layer = nn.Linear(self.rnn_unit, 256)
        self.projection_layer_2 = nn.Linear(256, self.output_dim)

    def _compute_sampling_threshold(self, batches_seen):
        return self.cl_decay_steps / (
                self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps))

    def forward(self, encoder_hidden_state, mask = None,labels=None, batches_seen=None):

        batch_size = encoder_hidden_state.size(1)

        go_symbol = torch.zeros((batch_size, self.num_node, self.output_dim),
                                device=device)
        decoder_hidden_state = encoder_hidden_state
        decoder_input = go_symbol

        outputs = []

        for t in range(self.horrizon):
            hidden_states = []
            output = decoder_input

            #noise
            distribution = torch.empty(output.shape).normal_(mean=0, std=0.01).to(device)
            output = output + distribution

            output = F.relu(self.input_projection(output))
            output = F.relu(self.input_projection_2(output))

            for layer_num, stgru_layer in enumerate(self.gwgru):

                next_hidden_state = stgru_layer(output, decoder_hidden_state[layer_num])
                hidden_states.append(next_hidden_state)

                # if self.num_rnn_layers > 2 and 1 <= layer_num < (self.num_rnn_layers - 1):
                #     output = output + next_hidden_state
                #     shape = output.shape
                #     output = torch.reshape(output, (-1, self.rnn_unit))
                #     output = self.batch_norm[layer_num - 1](output)
                #     output = torch.reshape(output,shape)
                # else:
                #     output = next_hidden_state
                output = output + next_hidden_state
                # shape = output.shape
                # output = torch.reshape(output, (-1, self.rnn_unit))
                # output = F.relu(self.batch_norm[layer_num](output))
                output = F.relu(output)
                # output = torch.reshape(output, shape)
            #normalization
            # output = output + output_res
            # output = torch.reshape(output, (-1, self.rnn_unit))
            # output = self.normalization(output)
            # output = torch.reshape(output, (batch_size, self.num_node, self.rnn_unit))

            decoder_output = F.relu(self.projection_layer(output))
            decoder_output = self.projection_layer_2(decoder_output)
            # decoder_output = self.projection_layer(output)

            decoder_hidden_state = torch.stack(hidden_states)
            decoder_input = decoder_output

            outputs.append(decoder_output)
            if labels is not None:
                # c = np.random.uniform(0, 1)
                # if c < self._compute_sampling_threshold(batches_seen):
                #     decoder_input = labels[t]
                decoder_input = labels[t] * mask[t] + decoder_input * (1 - mask[t])
        outputs = torch.stack(outputs)

        return outputs

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
        inputs = inputs.permute(1,0,2,3)
        mask = mask.permute(1,0,2,3)

        if labels is not None:
            labels = labels.permute(1,0,2,3)
        encoder_hidden_state = self.encoder_model(inputs, mask)

        outputs = self.decoder_model(encoder_hidden_state,mask, labels, batches_seen=batches_seen)
        # output:[horrizon, B, N, output_dim]
        outputs = outputs.permute(1,0,2,3)
        return outputs

if __name__ == '__main__':
    adj = np.load('../PEMS(M)/dataset/W_228.npy')
    adj = torch.from_numpy(adj).to(device).to(torch.float)
    print(adj.dtype)

    model = Model(1, [adj], 32, 228, 4, 1, 12,12).to(device)
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
    #     if(para.requires_grad):
    #         print(name)
    #         print(para)
    #         print(para.requires_grad)
    #         print('________________')