import torch.nn as nn
import numpy as np
import torch
from torch.autograd import Variable

def _sort(_2dtensor, lengths, descending=True):
    sorted_lengths, order = lengths.sort(descending=descending)
    _2dtensor_sorted_by_lengths = _2dtensor[order]
    return _2dtensor_sorted_by_lengths, order


class CharEmbed(nn.Module):
    def __init__(self, alphabetSize, charEmbedDim, charLstmDim, bidirectional):
        super(CharEmbed, self).__init__()
        self.charEmbedding = nn.Embedding(alphabetSize, charEmbedDim)  # setin params traiN"!!

        # Performing LSTM encoding on the character embeddings
        # if params.charMode == 'LSTM':

        #init embed
        if bidirectional:
                charLstmDim = charLstmDim // 2
        #bias=true??
        self.charLstm = nn.LSTM(input_size=charEmbedDim, hidden_size=charLstmDim, num_layers=1, bidirectional=bidirectional, batch_first=True)
        #test fropout
        #initt lstm vs other
        # init_lstm(self.CharLstm)

    def forward(self, sentence, charInfo):
        charsToIdx = charInfo

        word_lengths = charsToIdx.gt(0).sum(1)  # actual word lengths
        sorted_padded, order = _sort(charsToIdx, word_lengths)
        embedded = self.charEmbedding(sorted_padded)

        word_lengths_copy = word_lengths.clone()
        word_lengths_copy[word_lengths == 0] = 1
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, word_lengths_copy[order], True) #pack char sequence
        packed_output, _ = self.charLstm(packed)

        output, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_output, True) #unpack output
        _, reverse_sort_order = torch.sort(order, dim=0)
        output = output[reverse_sort_order]

        indices_of_lasts = (word_lengths_copy - 1).unsqueeze(1).expand(-1, output.shape[2]).unsqueeze(1)
        output = output.gather(1, indices_of_lasts).squeeze()
        output[word_lengths == 0] = 0
        return output




        #
        # #print(charsToIdx)
        # chars_embeds = self.charEmbedding(charsToIdx)
        # chars_embeds = chars_embeds.transpose(0, 1)
        # print(charToIdxLengths)
        # packed = torch.nn.utils.rnn.pack_padded_sequence(chars_embeds, charToIdxLengths)
        # lstmOut, _ = self.charLstm(packed)
        # outputs, outputLengths = torch.nn.utils.rnn.pad_packed_sequence(lstmOut)
        # outputs = outputs.transpose(0, 1)
        # chars_embeds_temp = Variable(torch.FloatTensor(torch.zeros((outputs.size(0), outputs.size(2)))))
        # if self.use_gpu:
        #     chars_embeds_temp = chars_embeds_temp.cuda()
        # for i, index in enumerate(outputLengths):
        #     chars_embeds_temp[i] = torch.cat((outputs[i, index - 1, :self.char_lstm_dim], outputs[i, 0, self.char_lstm_dim:]))
        # chars_embeds = chars_embeds_temp.clone()
        # for i in range(chars_embeds.size(0)):
        #     chars_embeds[d[i]] = chars_embeds_temp[i]
        #
        # return chars_embeds





