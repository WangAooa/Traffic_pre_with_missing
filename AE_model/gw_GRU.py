import numpy as np
import torch
import torch.nn as nn

from AE_model.gw_GRU_cell import *
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:2") if torch.cuda.is_available() else torch.device("cpu")

###used the unit model : GCN is embeded in GRU (2 layer)

class EncoderModel(nn.Module):
    def __init__(self, intput_dim, adj, rnn_unit, num_node, num_rnn_layers):
        super(EncoderModel, self).__init__()
        self.input_dim = intput_dim
        self.adj = adj
        self.rnn_unit = rnn_unit
        self.num_node = num_node
        self.num_rnn_layers = num_rnn_layers
        self.gwgru = nn.ModuleList([gw_GUR_cell(self.rnn_unit * 2, self.rnn_unit, self.adj,self.num_node),
                                    gw_GUR_cell(2*self.rnn_unit, self.rnn_unit, self.adj,self.num_node)])

        self.gwgru_reverse = nn.ModuleList([gw_GUR_cell(self.rnn_unit * 2, self.rnn_unit, self.adj, self.num_node),
                                            gw_GUR_cell(self.rnn_unit * 2, self.rnn_unit, self.adj, self.num_node)])

        # self.normalization = nn.BatchNorm1d(self.rnn_unit)
        # self.normalization_reverse = nn.BatchNorm1d(self.rnn_unit)

        # self.input_projection = nn.Linear(in_features=self.input_dim * 2, out_features= self.rnn_unit)
        self.input_projection = nn.Linear(in_features=self.input_dim * 2, out_features=256)
        self.input_projection_2 = nn.Linear(256,self.rnn_unit)

        # self.gwgru = nn.ModuleList([gw_GUR_cell(self.rnn_unit * 2, self.rnn_unit, self.adj, self.num_node)])

        # self.gwgru_reverse = nn.ModuleList([gw_GUR_cell(self.rnn_unit * 2, self.rnn_unit, self.adj, self.num_node)])
        #
        self.linear_dense_layer = nn.Linear(2*self.rnn_unit, 256)
        self.linear_dense_layer_2 = nn.Linear(256, self.rnn_unit)
        # self.stgru = nn.ModuleList([STGRU_cell(self.input_dim + self.rnn_unit, self.rnn_unit, self.adj, self.num_node)]
        #                             )

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
            output = self.input_projection_2(output)

            for layer_num, gwgru_layer in enumerate(self.gwgru):
                next_hidden_state = gwgru_layer(output, hidden_state[layer_num])

                hidden_states.append(next_hidden_state)
                output = next_hidden_state

                #normalization
                # output = next_hidden_state + output
                # output = torch.reshape(output, (-1,self.rnn_unit))
                # output = self.normalization(output)
                # output = torch.reshape(output, (batch_size,self.num_node,self.rnn_unit))

            hidden_state = torch.stack(hidden_states)

        inputs_reverse = torch.flip(inputs, [0])

        for t in range(time_step):
            hidden_states = []
            output = inputs_reverse[t]

            output = F.relu(self.input_projection(output))
            output = self.input_projection_2(output)

            for layer_num, gwgru_layer in enumerate(self.gwgru_reverse):
                next_hidden_state = gwgru_layer(output, hidden_state_reverse[layer_num])

                hidden_states.append(next_hidden_state)
                output = next_hidden_state

                # normalization
                # output = next_hidden_state + output
                # output = torch.reshape(output, (-1, self.rnn_unit))
                # output = self.normalization(output)
                # output = torch.reshape(output, (batch_size, self.num_node, self.rnn_unit))

            hidden_state_reverse = torch.stack(hidden_states)

        hidden_state = torch.cat([hidden_state, hidden_state_reverse], dim=3)
        hidden_state = F.relu(self.linear_dense_layer(hidden_state))
        hidden_state = self.linear_dense_layer_2(hidden_state)
        # hidden_states = []
        # output = inputs
        # for layer_num, stgru_layer in enumerate(self.gwgru):
        #     next_hidden_state = stgru_layer(output, hidden_state[layer_num])
        #
        #     hidden_states.append(next_hidden_state)
        #     output = next_hidden_state

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
        self.stgru_layers = nn.ModuleList([gw_GUR_cell(2*self.rnn_unit, self.rnn_unit, self.adj,self.num_node),
                                           gw_GUR_cell(2*self.rnn_unit, self.rnn_unit, self.adj,self.num_node)])

        # self.normalization = nn.BatchNorm1d(self.rnn_unit)

        self.input_projection = nn.Linear(in_features=self.output_dim, out_features=256)
        self.input_projection_2 = nn.Linear(256, self.rnn_unit)

        # self.stgru_layers = nn.ModuleList(
        #     [gw_GUR_cell(self.rnn_unit * 2, self.rnn_unit, self.adj, self.num_node)])

        # self.stgru_layers = nn.ModuleList(
        #     [STGRU_cell(self.output_dim + self.rnn_unit, self.rnn_unit, self.adj, self.num_node)])

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
            output_res = self.input_projection_2(output)

            output = output_res
            for layer_num, stgru_layer in enumerate(self.stgru_layers):

                next_hidden_state = stgru_layer(output, decoder_hidden_state[layer_num])
                hidden_states.append(next_hidden_state)
                output = next_hidden_state

                # normalization
                # output = next_hidden_state + output
                # output = torch.reshape(output, (-1, self.rnn_unit))
                # output = self.normalization(output)
                # output = torch.reshape(output, (batch_size, self.num_node, self.rnn_unit))
            #normalization
            output = output + output_res
            output = torch.reshape(output, (-1, self.rnn_unit))
            output = self.normalization(output)
            output = torch.reshape(output, (batch_size, self.num_node, self.rnn_unit))

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


        # for layer_num, stgru_layer in enumerate(self.stgru_layers):
        #     next_hidden_state = stgru_layer(output, hidden_state[layer_num])
        #     hidden_states.append(next_hidden_state)
        #     output = next_hidden_state

        #projected = self.projection_layer(output.view(-1, self.rnn_unit))
        #output = projected.view(-1, self.num_node, self.output_dim)

        # output = self.projection_layer(output)

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
        self.linear_dense_layer = nn.Linear(in_features=2*self.rnn_unit, out_features=self.rnn_unit)

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

        inputs_reverse = torch.flip(inputs, [0])
        encoder_hidden_state_reverse = None
        for t in range(self.seq_len):
            _, encoder_hidden_state_reverse = self.encoder_model_reverse(inputs_reverse[t], encoder_hidden_state_reverse)

        encoder_hidden_state = torch.cat([encoder_hidden_state,encoder_hidden_state_reverse], dim=2)
        encoder_hidden_state = self.linear_dense_layer(encoder_hidden_state)

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
            if self.training :
                c = np.random.uniform(0, 1)
                if c < self._compute_sampling_threshold(batches_seen):
                    decoder_input = labels[t]
        outputs = torch.stack(outputs)
        #output:[horrizon, B, N, output_dim]
        return outputs

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

        outputs = self.decoder_model(encoder_hidden_state,mask, inputs, batches_seen=batches_seen)
        # output:[horrizon, B, N, output_dim]
        outputs = outputs.permute(1,0,2,3)
        return outputs

if __name__ == '__main__':
    adj = np.load('../PEMS(M)/dataset/W_228.npy')
    adj = torch.from_numpy(adj).to(device).to(torch.float)
    print(adj.dtype)

    model = Model(1, [adj], 32, 228, 2, 1, 12,12).to(device)
    print(model)
    x = torch.zeros((32,12,228,1)).to(device)
    x_mask = torch.zeros((32,12,228,1)).to(device)
    time_lag = torch.zeros((32,12,228,1)).to(device)

    result = model(x,x_mask)
    print(result.size())