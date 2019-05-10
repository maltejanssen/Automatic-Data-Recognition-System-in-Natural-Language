import os
import random
import torch
from torch.autograd import Variable
import numpy as np
from util import Params
from nltk import sent_tokenize




class DataLoader(object):
    """ Loads and stores data with their mappings to indices.
    """


    def __init__(self, path, params, encoding="utf-8"):
        """ Loads vocabulary, tags and parameters of dataset

        :param str path: path to data directory
        :param Params params: hyperparameters for training
        """
        self.encoding = encoding
        jasonPath = "Data/dataset_params.json"
        assert os.path.isfile(jasonPath), "No json file found at {}, run build_vocab.py".format(jasonPath)
        self.datasetParams = Params(jasonPath)

        VocabPath = "Data/words.txt"
        self.vocab = {}
        with open(VocabPath, encoding=encoding) as f:
            for idx, word in enumerate(f.read().splitlines()):
                self.vocab[word] = idx

        self.unkInd = self.vocab[self.datasetParams.unk_word]
        self.padInd = self.vocab[self.datasetParams.pad_word]

        tagsPath = os.path.join(path, 'tags.txt')
        self.tagMap = {}
        self.idxToTag = {}
        with open(tagsPath) as f:
            for idx, tag in enumerate(f.read().splitlines()):
                self.tagMap[tag] = idx
                self.idxToTag[idx] = tag

        params.update(jasonPath)


    def getidxToTag(self):
        """ idxToTag getter

        :return: idxToTag lookup list
        """
        return self.idxToTag


    def getpadInd(self):
        """ padInd getter

        :return: index of padding string
        """
        return self.padInd

    def loadSentences(self, sentence):
        """ translates Sentences into word, indicie mapping

        :param list sentencesList: List of strings(sentences)
        :return list sentences: indice map
        """
        returnSentences = []
        words = []

        s = [self.vocab[token] if token in self.vocab
             else self.unkInd
             for token in sentence.split(' ')]
        returnSentences.append(s)

        for token in sentence.split(" "):
            words.append(token)




        return words, returnSentences


    def load_sentences_labels(self, sentencesFile, labelsFile):
        """ Loads sentences and labels , maps tokens and tags to their indices

        :param sentencesFile: file containing sentences
        :param labelsFile: file containing labels

        :return: dictionary containing loaded data
        """
        sentences = []
        labels = []
        data = {}


        with open(sentencesFile, encoding=self.encoding) as f:
            for sentence in f.read().splitlines():
                # replace each token by its index if it is in vocab
                # else use index of unknown word replacement
                s = [self.vocab[token] if token in self.vocab
                     else self.unkInd
                     for token in sentence.split(' ')]
                sentences.append(s)

        with open(labelsFile) as f:
            for sentence in f.read().splitlines():
                # replace each label by its index
                l = [self.tagMap[label] if label in self.tagMap else "wtf" for label in sentence.split(' ')]
                #TODO handle unexpected tags
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

        for split in ["train", "val", "test"]:
            if split in types:
                sentencesFile = os.path.join(path, split, "sentences.txt")
                labelsFile = os.path.join(path, split, "labels.txt")
                data[split] = self.load_sentences_labels(sentencesFile, labelsFile)
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

            batchSentences = [data['sentences'][idx] for idx in order[batch * params.batch_size:(batch + 1) * params.batch_size]]
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