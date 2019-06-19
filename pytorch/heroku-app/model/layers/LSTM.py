import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence,pack_padded_sequence
#from .model_utils import prepare_pack_padded_sequence
import numpy as np
import torch



class BILSTM(nn.Module):
    def __init__(self, embedDim, lstmDim, bidirectional, numberOFTags, charLstmDim=None, dropout=False, crf=False):

        super(BILSTM, self).__init__()
        self.crf = crf
        self.performDropout = dropout
        self.dropout = nn.Dropout(0.5)
        if charLstmDim:
            embedDim = embedDim + charLstmDim
        self.lstmDim = lstmDim
        if bidirectional:
            self.lstmDim = lstmDim // 2 #//2
            #embedDim = embedDim * 2 #only multiply second???
        self.lstm = nn.LSTM(embedDim, self.lstmDim, batch_first=True, bidirectional=bidirectional)

        #self.init_lstm(self.lstm)  # vs no ini

        # vs
        # if bidirectional:
        #     self.linear = nn.Linear(in_features=hidenDim * 2, out_features=numberOFTags)
        # else:
        #     self.linear = nn.Linear(in_features=hidenDim, out_features=numberOFTags)
        self.linear = nn.Linear(in_features=lstmDim, out_features=numberOFTags)

        #nn.init.xavier_uniform(self.linear.weight)
        # #vs
        # torch.nn.init.normal_(self.linear.weight, mean=0, std=1)
        # #vs
        # torch.nn.init.constant_(self.linear.weight, 0)
        # #vs
        # torch.nn.init.uniform_(self.linear.weight, a=0, b=1)



    def init_lstm(self, input_lstm):
        """
        Initialize lstm

        PyTorch weights parameters:

            weight_ih_l[k]: the learnable input-hidden weights of the k-th layer,
                of shape `(hidden_size * input_size)` for `k = 0`. Otherwise, the shape is
                `(hidden_size * hidden_size)`

            weight_hh_l[k]: the learnable hidden-hidden weights of the k-th layer,
                of shape `(hidden_size * hidden_size)`
        """

        # Weights init for forward layer
        for ind in range(0, input_lstm.num_layers):
            ## Gets the weights Tensor from our model, for the input-hidden weights in our current layer
            weight = eval('input_lstm.weight_ih_l' + str(ind))

            # Initialize the sampling range
            sampling_range = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))

            # Randomly sample from our samping range using uniform distribution and apply it to our current layer
            nn.init.uniform(weight, -sampling_range, sampling_range)

            # Similar to above but for the hidden-hidden weights of the current layer
            weight = eval('input_lstm.weight_hh_l' + str(ind))
            sampling_range = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
            nn.init.uniform(weight, -sampling_range, sampling_range)

        # We do the above again, for the backward layer if we are using a bi-directional LSTM (our final model uses this)
        if input_lstm.bidirectional:
            for ind in range(0, input_lstm.num_layers):
                weight = eval('input_lstm.weight_ih_l' + str(ind) + '_reverse')
                sampling_range = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
                nn.init.uniform(weight, -sampling_range, sampling_range)
                weight = eval('input_lstm.weight_hh_l' + str(ind) + '_reverse')
                sampling_range = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
                nn.init.uniform(weight, -sampling_range, sampling_range)

        # Bias initialization steps

        # We initialize them to zero except for the forget gate bias, which is initialized to 1
        if input_lstm.bias:
            for ind in range(0, input_lstm.num_layers):
                bias = eval('input_lstm.bias_ih_l' + str(ind))

                # Initializing to zero
                bias.data.zero_()

                # This is the range of indices for our forget gates for each LSTM cell
                bias.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1

                # Similar for the hidden-hidden layer
                bias = eval('input_lstm.bias_hh_l' + str(ind))
                bias.data.zero_()
                bias.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1

            # Similar to above, we do for backward layer if we are using a bi-directional LSTM
            if input_lstm.bidirectional:
                for ind in range(0, input_lstm.num_layers):
                    bias = eval('input_lstm.bias_ih_l' + str(ind) + '_reverse')
                    bias.data.zero_()
                    bias.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1
                    bias = eval('input_lstm.bias_hh_l' + str(ind) + '_reverse')
                    bias.data.zero_()
                    bias.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1

    def forward(self,input):
        # embeddings_packed: (batch_size*time_steps, embedding_dim)

        # inputs, length, desorted_indice = prepare_pack_padded_sequence(inputs, length)
        # embeddings_packed = pack_padded_sequence(inputs, length, batch_first=True)
        #output, _ = self.lstm(embeddings_packed)

        s, _ = self.lstm(input)
        #output, _ = pad_packed_sequence(output, batch_first=True)
        #output = output[desorted_indice]

        if self.performDropout:
            s = self.dropout(s)
        # output = F.tanh(output)
        # make the Variable contiguous in memory (a PyTorch artefact)
        s = s.contiguous()
        # reshape the Variable so that each row contains one token
        # print(s.view(-1, s.shape[2]//2))
        # print(s.view(-1, s.shape[2]//2).size())
        if not self.crf:
            s = s.view(-1, s.shape[2]) # dim: batch_size*seq_len x lstm_hidden_dim  #test without this
        #s = s.view(len(input), self.lstmDim)
       # s = s.view(len(input), self.lstmDim)

        logit = self.linear(s)
        return logit, self.lstmDim

