import os
import random
import torch
from torch.autograd import Variable
import numpy as np
from .util import Params



class DataLoader(object):
    """ Loads and stores data with their mappings to indices.
    """

    def __init__(self, path, params):
        """ Loads vocabulary, tags and parameters of dataset

        :param str path: path to data directory
        :param Params params: hyperparameters for training
        """
        jasonPath = "Data/dataset_params.json"
        assert os.path.isfile(jasonPath), "No json file found at {}, run build_vocab.py".format(jasonPath)
        self.datasetParams = Params(jasonPath)

        VocabPath = "Data/words.txt"
        self.vocab = {}
        with open(VocabPath) as f:
            for idx, word in enumerate(f.read().splitlines()):
                self.vocab[word] = idx

        self.unkInd = self.vocab[self.datasetParams.unk_word]
        self.padInd = self.vocab[self.datasetParams.pad_word]

        tagsPath = os.path.join(path, 'tags.txt')
        self.tagMap = {}
        with open(tagsPath) as f:
            for idx, tag in enumerate(f.read().splitlines()):
                self.tagMap[tag] = idx

        params.update(jasonPath)

    def load_sentences_labels(self, FSentences, FLabels):
        """ Loads sentences and labels , maps tokens and tags to their indices

        :param FSentences: file containing sentences
        :param FLabels: file containing labels
        :return: dictionary containing loaded data
        """
        sentences = []
        labels = []
        data = {}

        with open(FSentences) as f:
            for sentence in f.read().splitlines():
                # replace each token by its index if it is in vocab
                # else use index of UNK_WORD
                s = [self.vocab[token] if token in self.vocab
                     else self.unk_ind
                     for token in sentence.split(' ')]
                sentences.append(s)

        with open(FLabels) as f:
            for sentence in f.read().splitlines():
                # replace each label by its index
                l = [self.tag_map[label] for label in sentence.split(' ')]
                labels.append(l)

        assert len(labels) == len(sentences)
        for i in range(len(labels)):
            assert len(labels[i]) == len(sentences[i])

        data['sentences'] = sentences
        data['labels'] = labels
        data['size'] = len(sentences)
        return data


    def readData(self, path, types):
        """ loads data of given types (eg.: train, val, test)

        :param list types: containing one or more of 'train', 'val', 'test'
        :param path: path to dataset
        :return: dictionary containing data( sentences and labels) for each type
        """
        data = {}

        for split in ['train', 'val', 'test']:
            if split in types:
                sentences_file = os.path.join(path, split, "sentences.txt")
                labels_file = os.path.join(path, split, "labels.txt")
                data[split] = self.load_sentences_labels(sentences_file, labels_file)
        return data


    def batchGenerator(self, data, params, shuffle=False):
        """ creates a Generator for generating batches of data

        :param dict data: data dictionary created by loadData (with keys "sentences", "labels", "size)
        :param params: hyperparameters for training
        :param bool shuffle: should data be shuffled?
        :return Variable batchSentences: sentences of batch
        :return Variable batchLabels: labels of batch
        """
        # make a list that decides the order in which we go over the data- this avoids explicit shuffling of data
        order = list(range(data['size']))
        if shuffle:
            random.seed(230)
            random.shuffle(order)

        amountOfBatches = (data['size'] + 1) // params.batch_size
        for batch in range(amountOfBatches):
            batchSentences = [data['data'][idx] for idx in order[batch * params.batch_size:(batch + 1) * params.batch_size]]
            batchTags = [data['labels'][idx] for idx in order[batch * params.batch_size:(batch + 1) * params.batch_size]]

            longestSentence = max([len(s) for s in batchSentences])

            # prepare a numpy array with the data, initialising the data with pad_ind and all labels with -1
            # initialising labels to -1 differentiates tokens with tags from PADding tokens
            batchData = self.padInd * np.ones((len(batchSentences), longestSentence))
            batchLabels = -1 * np.ones((len(batchSentences), longestSentence))

            for j in range(len(batchSentences)):
                sentenceLen = len(batchSentences[j])
                batchData[j][:sentenceLen] = batchSentences[j]
                batchLabels[j][:sentenceLen] = batchTags[j]

            batchData, batchLabels = torch.LongTensor(batchData), torch.LongTensor(batchLabels)

            # shift tensors to GPU if available
            if params.cuda:
                batchData, batchLabels = batchData.cuda(), batchLabels.cuda()

            # convert them to Variables to record operations in the computational graph
            batchData, batchLabels = Variable(batchData), Variable(batchLabels)

            yield batchData, batchLabels