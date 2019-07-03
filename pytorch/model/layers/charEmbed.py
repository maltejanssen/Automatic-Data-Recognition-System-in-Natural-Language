import torch.nn as nn
import torch
from torch.autograd import Variable


def _sort(_2dtensor, lengths, descending=True):
    sorted_lengths, order = lengths.sort(descending=descending)
    _2dtensor_sorted_by_lengths = _2dtensor[order]
    return _2dtensor_sorted_by_lengths, order


class CharEmbed(nn.Module):
    def __init__(self, alphabetSize, charEmbedDim, charLstmDim, bidirectional):
        super(CharEmbed, self).__init__()
        self.charEmbedding = nn.Embedding(alphabetSize, charEmbedDim)

        if bidirectional:
                charLstmDim = charLstmDim // 2
        self.charLstm = nn.LSTM(input_size=charEmbedDim, hidden_size=charLstmDim, num_layers=1, bidirectional=bidirectional, batch_first=True)


    def forward(self, input, charEmbed):


        wordLengths = charEmbed.gt(0).sum(1)  # actual word lengths
        sorted, order = _sort(charEmbed, wordLengths)  #pack_padded_sequence expects sorted

        embedded = self.charEmbedding(sorted)

        wordLengthCopy = wordLengths.clone()
        wordLengthCopy[wordLengths == 0] = 1 #pack_padded_sequence needds all samples to have length >= 1
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, wordLengthCopy[order], batch_first=True) #pack char sequence
        packedOut, _ = self.charLstm(packed)

        output, _ = torch.nn.utils.rnn.pad_packed_sequence(packedOut, True) #unpack output
        _, reversedSort = torch.sort(order, dim=0)
        output = output[reversedSort]

        indicesLastDim = (wordLengthCopy - 1).unsqueeze(1).expand(-1, output.shape[2]).unsqueeze(1)
        output = output.gather(1, indicesLastDim).squeeze()

        output[wordLengths == 0] = 0 #reset lengths

        return output






