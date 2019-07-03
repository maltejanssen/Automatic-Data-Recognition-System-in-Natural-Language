import numpy as np
np.random.seed(200)
import torch
import torch.nn as nn
import torch.nn.functional as F

from .eval import accuracy, f1
from .layers.EmbedLayer import EmbedLayer
from .layers.LSTM import LSTM
from .layers.charEmbed import CharEmbed
from TorchCRF import CRF
from .loss import lossFn


class Net(nn.Module):
    """ at least Lstm net, optional: BiLSTM - CRF with char embedding

    """
    def __init__(self, params, embedWeights=None):
        """

        :param params: networks parameters, includes: model configuration and dataset info
        :param embedWeights: optional pre trained embedding weights
        """
        super(Net, self).__init__()

        # load optional parameters
        try:
            params.bidirectional
        except AttributeError as e:
            bidirectional = False
        else:
            bidirectional = True if params.bidirectional == "True" else False

        try:
            params.crf
        except AttributeError as e:
            self.useCrf = False
        else:
            self.useCrf = True if params.crf == "True" else False

        try:
            params.dropout
        except AttributeError as e:
            self.performDropout = False
        else:
            self.performDropout = True if params.dropout == "True" else False

        try: params.char_lstm_dim
        except AttributeError as e:
            print("param char_lstm_dim not defined in params file", "continuing without char embedding")
            self.doCharEmbed = False
            char_lstm_dim = None
        else:
            self.doCharEmbed = True
            self.charEmbedding = CharEmbed(alphabetSize=params.alphabet_size, charEmbedDim=params.char_embedding_dim,
                                            charLstmDim=params.char_lstm_dim, bidirectional=bidirectional)
            char_lstm_dim = params.char_lstm_dim

        self.embedding = EmbedLayer(embedWeights=embedWeights, vocabSize=params.vocab_size, embedDim=params.embedding_dim)

        self.dropout = nn.Dropout(0.5)
        self.lstm = LSTM(self.embedding.embedDim, params.lstm_hidden_dim, bidirectional, params.number_of_tags, charLstmDim=char_lstm_dim, dropout=self.performDropout, crf=self.useCrf)

        self.metrics = {
                        'f1': f1, #"accuracy" : accuracy,
                        }

        if self.useCrf:
            self.crflayer = CRF(params.number_of_tags)
        else:
            self.crflayer = None

        self.lossFN = lossFn


    def getLstmFeatures(self, input, charInfo=None):
        """ acts as pre-crf forward function

        :param input: input batch
        :param charInfo: char input from batch
        :return: linearised lstm features
        """
        # input dim: batch_size x seq_len


        embeds = self.embedding(input)  # dim: batch_size x seq_len x embedding_dim
        if self.doCharEmbed:
            try:
                assert charInfo is not None
            except AssertionError as e:
                e.args += ("no char embedding info provided", "aborting")
                raise
            charEmbeds = self.charEmbedding(input, charInfo)  # wrong dim
            batch_size = input.size(0)
            max_seq_len = input.size(1)

            charEmbeds = charEmbeds.reshape(batch_size, max_seq_len,
                                            -1)  # reshape to batch_size x seq_len x char_lstm_hidden_dim


            embeds = torch.cat([embeds, charEmbeds], -1)  # dim: batch_size x seq_len x embed_dim + char_lstm_hidden_dim
        if self.performDropout:
            embeds = self.dropout(embeds)

        logit, lstmDim = self.lstm(embeds)  # linear aslo done in lstm because dimension calculated there
        return logit


    def loss(self, outputs, goldLabels, mask=None):
        """ calculates cross entrophy loss for tokens

        :param feats: lstm features
        :param goldLabels: goldLabels of batch
        :param mask: non pad mask, needed for crf
        :return : average token cross entrophy loss
        """

        if self.useCrf:
            try:
                assert mask is not None
            except AssertionError as e:
                e.args += ("no mask provided", "aborting")
                raise
            num_tokens = int(torch.sum(mask).item())
            loss = self.crflayer.forward(outputs, goldLabels, mask)
            return -torch.sum(loss) / num_tokens

        else:
            return lossFn(outputs, goldLabels)


    def forward(self, input, charEmbed=None):
        feats = self.getLstmFeatures(input, charEmbed)
        if self.useCrf:
            return feats
        else:
            return F.log_softmax(feats, dim=1)








