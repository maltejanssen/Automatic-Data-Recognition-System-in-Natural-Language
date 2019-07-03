import numpy as np
import torch
import torch.nn as nn

class EmbedLayer(nn.Module):
    def __init__(self, embedWeights = None, vocabSize = None, embedDim=None):
        super(EmbedLayer, self).__init__()

        if embedWeights is not None:
            self.embedDim = len(embedWeights[0])
        else:
            self.embedDim = embedDim

        # the embedding takes as input the vocab_size and the embedding_dim
        vocabSize = vocabSize
        self.embedding = nn.Embedding(vocabSize, self.embedDim)
        if embedWeights is not None:
            self.embedding.weight.data = torch.Tensor(embedWeights)


    def createRndEmbedding(self, vocabSize, embedDim):
        emb = np.empty([vocabSize, embedDim])
        scale = np.sqrt(3.0 / embedDim)
        for index in range(vocabSize):
            emb[index,:] = np.random.uniform(-scale, scale, [1, embedDim])
        return emb

    def forward(self, inputs):
        output = self.embedding(inputs)
        #dim: batch_size x seq_len x embedding_dim

        return output
