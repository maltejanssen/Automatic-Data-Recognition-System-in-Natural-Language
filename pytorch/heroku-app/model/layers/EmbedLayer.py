import numpy as np
import torch
import torch.nn as nn

class EmbedLayer(nn.Module):
    def __init__(self, embedWeights = None, vocabSize = None, embedDim=None):
        super(EmbedLayer, self).__init__()
        #self.training = training
        if embedWeights is not None:
            self.embedDim = len(embedWeights[0])
        else:
            self.embedDim = embedDim

        # the embedding takes as input the vocab_size and the embedding_dim
        vocabSize = vocabSize
        self.embedding = nn.Embedding(vocabSize, self.embedDim)
        if embedWeights is not None:
            self.embedding.weight.data = torch.Tensor(embedWeights)
        #test performance of this
        else:
            sd = 0.1
            #self.embedding.weight.data = (torch.from_numpy(self.createRndEmbedding(vocabSize, self.embedDim)))
            #self.embedding.weight.data = torch.from_numpy(np.random.normal(0, scale=sd, size=[vocabSize, self.embedDim]))

            #self.embedding.weight.data.copy_(torch.from_numpy(self.createRndEmbedding(vocabSize+2, self.embedDim)))

        #test dropout
        #self.dropout = nn.Dropout(dropout_emb)

    def createRndEmbedding(self, vocabSize, embedDim):
        emb = np.empty([vocabSize, embedDim])
        scale = np.sqrt(3.0 / embedDim)
        for index in range(vocabSize):
            emb[index,:] = np.random.uniform(-scale, scale, [1, embedDim])
        return emb

    def forward(self, inputs):
        s = self.embedding(inputs)
        #dim: batch_size x seq_len x embedding_dim
        #x = self.dropout(x)
        return s
