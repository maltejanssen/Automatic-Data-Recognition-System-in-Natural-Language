import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def accuracy(outputs, labels):
    """ calculates accuracy, excludes paddings

    :param np.ndarray outputs: log softmax output of the model
    :param np.ndarray labels: indices of labels (-1 if padding) [0, 1, ... num_tag-1]
    :return float accuracy: in interval [0,1]
    """
    # flat vector conversion
    labels = labels.ravel()

    #mask to exclude padding tokens
    mask = (labels >= 0)

    predictions = np.argmax(outputs, axis=1)
    accuracy = np.sum((predictions==labels)/float(np.sum(mask)))
    return accuracy
#TODO define more functions such as f1 and accuracy for each label

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

        self.metrics = {
                        'accuracy': accuracy,

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

        # apply log softmax on each token's output (this is recommended over applying softmax
        # since it is numerically more stable)
        return F.log_softmax(s, dim=1)   # dim: batch_size*seq_len x num_tags


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
        labels = labels % outputs.shape[1]

        #num of non mask tokens
        num_tokens = int(torch.sum(mask).item()) #.data[0]

        # compute cross entropy loss for all tokens (except PADding tokens), by multiplying with mask.
        return -torch.sum(outputs[range(outputs.shape[0]), labels]*mask)/num_tokens





