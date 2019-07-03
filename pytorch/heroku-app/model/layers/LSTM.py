import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, embedDim, lstmDim, bidirectional, numberOFTags, charLstmDim=None, dropout=False, crf=False):

        super(LSTM, self).__init__()
        self.useCrf = crf
        self.performDropout = dropout
        self.dropout = nn.Dropout(0.5)
        if charLstmDim:
            embedDim = embedDim + charLstmDim
        self.lstmDim = lstmDim
        if bidirectional:
            self.lstmDim = lstmDim // 2
        self.lstm = nn.LSTM(embedDim, self.lstmDim, batch_first=True, bidirectional=bidirectional)


        self.linear = nn.Linear(in_features=lstmDim, out_features=numberOFTags)


    def forward(self,input):

        output, _ = self.lstm(input)

        if self.performDropout:
            output = self.dropout(output)
        # make the Variable contiguous in memory
        output = output.contiguous()

        if not self.useCrf:
            output = output.view(-1, output.shape[2]) # dim: batch_size*seq_len x lstm_hidden_dim  #test without this

        logit = self.linear(output)
        return logit, self.lstmDim

