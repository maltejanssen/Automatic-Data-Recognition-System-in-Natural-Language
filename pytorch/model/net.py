import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from seqeval.metrics import f1_score
from torch.nn import CrossEntropyLoss


def accuracy(outputLabels, goldLabels):
    """ calculates accuracy, excludes paddings

    :param np.ndarray outputLabels: log softmax output of the model
    :param np.ndarray goldLabels: indices of labels (-1 if padding) [0, 1, ... num_tag-1]
    :return float accuracy: in interval [0,1]
    """
    # #old -> doesnt work for f1
    # # flat vector conversion
    # goldLabels = goldLabels.ravel()
    # # mask to exclude padding tokens
    # mask = (goldLabels >= 0)
    # predictions = np.argmax(outputLabels, axis=1)
    # accuracy = np.sum((predictions == goldLabels) / float(np.sum(mask)))

    # flat vector conversion
    goldLabels = goldLabels.ravel()

    predictions = np.argmax(outputLabels, axis=1)
    #get all indices with pad token; cant delete in for loop
    idcs = []
    for idx, label in enumerate(goldLabels):
        if label == -1:
            idcs.append(idx)

    goldLabels = np.delete(goldLabels, idcs)
    predictions = np.delete(predictions, idcs)
    accuracy = np.sum((predictions == goldLabels) / len(predictions))

    return accuracy
#TODO define more functions such as f1 and accuracy for each label

def f1(outputLabels, goldLabels):
    goldLabels = goldLabels.ravel()
    predictions = np.argmax(outputLabels, axis=1)
    idcs = []
    for idx, label in enumerate(goldLabels):
        if label == -1:
            idcs.append(idx)
    goldLabels = np.delete(goldLabels, idcs)
    predictions = np.delete(predictions, idcs)

    tagsPath = "Data/tags.txt"

    #DataLoader ?? maybe cereate simplified version of Dataloader
    idxToTag = {}
    with open(tagsPath) as f:
        for idx, tag in enumerate(f.read().splitlines()):
            idxToTag[idx] = tag

    assert len(goldLabels) == len(predictions)
    goldLabelsTranslation = []
    predictionsTranslation = []
    for gold, pred in zip(goldLabels, predictions):
        goldLabelsTranslation.append(idxToTag[gold])
        predictionsTranslation.append(idxToTag[pred])

    f1Score = f1_score(goldLabelsTranslation, predictionsTranslation)
    #print(f1Score)
    return f1Score



class Net(nn.Module):
    """ neural network
    """

    def __init__(self, params):
        """
        We define an recurrent network that predicts the NER tags for each token in the sentence. The components
        required are:

        - an embedding layer: this layer maps each index in range(params.vocab_size) to a params.embedding_dim vector
        - lstm: applying the LSTM on the sequential input returns an output for each token in the sentence
        - fc: a fully connected layer that converts the LSTM output for each token to a distribution over NER tags

        Args:
            params: (Params) contains vocab_size, embedding_dim, lstm_hidden_dim
        """
        super(Net, self).__init__()

        # the embedding takes as input the vocab_size and the embedding_dim
        self.embedding = nn.Embedding(params.vocab_size, params.embedding_dim)

        # the LSTM takes as input the size of its input (embedding_dim), its hidden size
        # for more details on how to use it, check out the documentation
        self.lstm = nn.LSTM(params.embedding_dim, params.lstm_hidden_dim, batch_first=True)

        # the fully connected layer transforms the output to give the final output layer
        self.fc = nn.Linear(params.lstm_hidden_dim, params.number_of_tags)

        #self.dropout =  nn.Dropout()

        self.metrics = {
                        "accuracy" : accuracy,
                        'f1': f1,
                        }
        
    def forward(self, s):
        """

        :param s: batch of sentences
        :return Variable out:  log probabilities of tokens for each token  of each sentence
        """
        """
        This function defines how we use the components of our network to operate on an input batch.

        Args:
            s: (Variable) contains a batch of sentences, of dimension batch_size x seq_len, where seq_len is
               the length of the longest sentence in the batch. For sentences shorter than seq_len, the remaining
               tokens are PADding tokens. Each row is a sentence with each element corresponding to the index of
               the token in the vocab.

        Returns:
            out: (Variable) dimension batch_size*seq_len x num_tags with the log probabilities of tokens for each token
                 of each sentence.

        Note: the dimensions after each step are provided
        """
        #                                -> batch_size x seq_len
        # apply the embedding layer that maps each token to its embedding
        s = self.embedding(s)            # dim: batch_size x seq_len x embedding_dim

        # run the LSTM along the sentences of length seq_len
        s, _ = self.lstm(s)              # dim: batch_size x seq_len x lstm_hidden_dim

        # make the Variable contiguous in memory (a PyTorch artefact)
        s = s.contiguous()

        # reshape the Variable so that each row contains one token
        s = s.view(-1, s.shape[2])       # dim: batch_size*seq_len x lstm_hidden_dim

        # apply the fully connected layer and obtain the output (before softmax) for each token
        s = self.fc(s)                   # dim: batch_size*seq_len x num_tags

        #s = self.dropout(s)

        # apply log softmax on each token's output (this is recommended over applying softmax
        # since it is numerically more stable)
        return F.log_softmax(s, dim=1)   # dim: batch_size*seq_len x num_tags
        #return s


    def lossFn(self, outputs, labels):
        """

        :param Variable outputs: log softmax output of the model
        :param Variable labels: indices of labels (-1 if padding) [0, 1, ... num_tag-1]
        :return Variable loss:  cross entropy loss for all tokens in the batch
        """
        # reshape labels to give a flat vector
        labels = labels.view(-1)
        # remove padding
        mask = (labels >= 0).float()



        # covert padding into positive, because of negative indexing errors -> but ignore with mask
        #labels = labels % outputs.shape[1]

        #num of non mask tokens
        num_tokens = int(torch.sum(mask).item()) #.data[0]
        # print(torch.sum(outputs[range(outputs.shape[0]), labels]*mask)/num_tokens)
        # print(-torch.sum(outputs[range(outputs.shape[0]), labels]*mask)/num_tokens)
        # compute cross entropy loss for all tokens (except PADding tokens), by multiplying with mask.
        return -torch.sum(outputs[range(outputs.shape[0]), labels]*mask)/num_tokens
        # numLabels = 13
        # active_loss = mask.view(-1) == 1
        # loss_fct = CrossEntropyLoss()
        #
        # active_logits = outputs.view(-1, numLabels)[active_loss]
        #
        # active_labels = labels.view(-1)[active_loss]
        # loss = loss_fct(active_logits, active_labels)
        #
        # return loss





