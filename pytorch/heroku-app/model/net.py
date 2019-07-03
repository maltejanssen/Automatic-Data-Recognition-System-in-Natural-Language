import numpy as np
np.random.seed(200)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss
import os
from .eval import accuracy, f1
from .layers.EmbedLayer import EmbedLayer
from .layers.LSTM import LSTM
from .layers.charEmbed import CharEmbed
from .layers.Crf import CRF





def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()



# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

START_TAG = "<START>"
STOP_TAG = "<STOP>"

class Net(nn.Module):

    def __init__(self, params, embedWeights=None):
        super(Net, self).__init__()
        self.cuda = params.cuda
        try: params.bidirectional
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
        self.lstm = LSTM(self.embedding.embedDim, params.lstm_hidden_dim, bidirectional, params.number_of_tags + 2, charLstmDim=char_lstm_dim, dropout=self.performDropout)

        # if self.useCrf:
        #     print("useCRF")
        #     self.crf = CRF(params.tagMap, self)

        self.metrics = {
                        "accuracy" : accuracy,
                        'f1': f1,
                        }
        self.cuda = params.cuda
        if self.useCrf:
            tag_to_ix = params.tagMap
            self.numTags = len(tag_to_ix)
            # Matrix of transition parameters.  Entry i,j is the score of
            # transitioning *to* i *from* j.
            random =  torch.randn(self.numTags, self.numTags) if not params.cuda\
                else torch.cuda.FloatTensor(self.numTags, self.numTags).normal_()
            self.transitions = nn.Parameter(random)


            # These two statements enforce the constraint that we never transfer
            # to the start tag and we never transfer from the stop tag
            self.tag_to_ix = tag_to_ix
            self.transitions.data[self.tag_to_ix[START_TAG], :] = -10000
            self.transitions.data[:, self.tag_to_ix[STOP_TAG]] = -10000


    def getLstmFeatures(self, input, charInfo=None):
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

            # embeds = torch.cat((embeds, charEmbeds), 1)
            embeds = torch.cat([embeds, charEmbeds], -1)  # dim: batch_size x seq_len x embed_dim + char_lstm_hidden_dim
        if self.performDropout:
            embeds = self.dropout(embeds)
        # embeds = embeds.unsqueeze(1)
        # embeds = self.dropout(embeds)
        # embeds = embeds.unsqueeze(1)
        logit, lstmDim = self.lstm(embeds)  # linear aslo done in lstm because dimension calculated there
        return logit



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

    def _forward_alg(self, feats):
        init_alphas = torch.Tensor(1, self.numTags).fill_(-10000.)
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.
        forward_var = autograd.Variable(init_alphas)
        if self.cuda:
            forward_var = forward_var.cuda()
        for feat in feats:
            emit_score = feat.view(-1, 1)
            tag_var = forward_var + self.transitions + emit_score
            max_tag_var, _ = torch.max(tag_var, dim=1)
            tag_var = tag_var - max_tag_var.view(-1, 1)
            forward_var = max_tag_var + torch.log(torch.sum(torch.exp(tag_var), dim=1)).view(1, -1) # ).view(1, -1)
        terminal_var = (forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]).view(1, -1)
        alpha = log_sum_exp(terminal_var)
        # Z(x)
        return alpha


        # # Do the forward algorithm to compute the partition function
        # init_alphas = torch.full((1, self.numTags), -10000.)
        # # START_TAG has all of the score.
        # init_alphas[0][self.tag_to_ix[START_TAG]] = 0.
        #
        # # Wrap in a variable so that we will get automatic backprop
        # forward_var = init_alphas
        #
        # # Iterate through the sentence
        # for feat in feats:
        #     alphas_t = []  # The forward tensors at this timestep
        #     for next_tag in range(self.numTags):
        #         # broadcast the emission score: it is the same regardless of
        #         # the previous tag
        #         emit_score = feat[next_tag].view(
        #             1, -1).expand(1, self.numTags)
        #         # the ith entry of trans_score is the score of transitioning to
        #         # next_tag from i
        #         trans_score = self.transitions[next_tag].view(1, -1)
        #         # The ith entry of next_tag_var is the value for the
        #         # edge (i -> next_tag) before we do log-sum-exp
        #         next_tag_var = forward_var + trans_score + emit_score
        #         # The forward variable for this tag is log-sum-exp of all the
        #         # scores.
        #         alphas_t.append(log_sum_exp(next_tag_var).view(1))
        #     forward_var = torch.cat(alphas_t).view(1, -1)
        # terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        # alpha = log_sum_exp(terminal_var)
        # return alpha

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        #score = torch.zeros(1)#

        # flatten
        tags = tags.view(-1)
        r = torch.LongTensor(range(feats.size()[0]))
        if self.cuda:
            r = r.cuda()
            #start = torch.cuda.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long)
            pad_start_tags = torch.cat([torch.cuda.LongTensor([self.tag_to_ix[START_TAG]]), tags])
            pad_stop_tags = torch.cat([tags, torch.cuda.LongTensor([self.tag_to_ix[STOP_TAG]])])
        else:
            #start = torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long)
            pad_start_tags = torch.cat([torch.LongTensor([self.tag_to_ix[START_TAG]]), tags])
            pad_stop_tags = torch.cat([tags, torch.LongTensor([self.tag_to_ix[STOP_TAG]])])


        #tags = torch.cat([start, tags])
        # for i, feat in enumerate(feats):
        #     score = score + \
        #             self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        # score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]

        score = torch.sum(self.transitions[pad_stop_tags, pad_start_tags]) + torch.sum(feats[r, tags])
        return score

    def viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.numTags), -10000.)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.numTags):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]  # Sanity check
        best_path.reverse()
        # print(path_score)
        # print(best_path)
        return path_score, best_path

    def neg_log_likelihood(self, feats, tags):
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score


    def forward(self, input, charInfo=None, eval=False):
        print(input)
        feats = self.getLstmFeatures(input, charInfo)
        # print("lsmsize")
        # print(logit.size())

        if self.useCrf:
            return feats

        else:
            return F.log_softmax(feats, dim=1)



