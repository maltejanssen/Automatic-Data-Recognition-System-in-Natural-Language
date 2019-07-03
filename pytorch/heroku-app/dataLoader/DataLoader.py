import os
import random
import torch
import nltk
from torch.autograd import Variable
from utils.util import Params
import numpy as np
np.random.seed(200)



nltk.download('punkt')


class DataLoader(object):
    """ Loads and stores data with their mappings to indices.
    """


    def __init__(self, path, params, encoding="utf-8", embedPath=None): #getr´rid of embed dim
        """ Loads vocabulary, tags and parameters of dataset

        :param str path: path to data directory
        :param Params params: hyperparameters for training
        """
        try:
            params.char_lstm_dim
        except AttributeError as e:
            self.loadCharEmbed = False
        else:
            self.loadCharEmbed = True

        self.encoding = encoding
        jasonPath = "Data/dataset_params.json"
        assert os.path.isfile(jasonPath), "No json file found at {}, run build_vocab.py".format(jasonPath)
        self.datasetParams = Params(jasonPath)

        VocabPath = "Data/words.txt"
        self.vocab = {}
        self.idxToWord = {}
        self.charMap = {}
        words = []
        with open(VocabPath, encoding=encoding) as f:
            for idx, word in enumerate(f.read().splitlines()):
                self.vocab[word] = idx
                self.idxToWord[idx] = word
                words.append(word)
        chars = set([w_i for w in words for w_i in w])

        self.charMap = {c: i for i, c in enumerate(chars)}



        lenChars = len(self.charMap)
        self.datasetParams.alphabet_size = lenChars
        self.datasetParams.charMap = self.charMap

        self.embeddings = None
        if embedPath is not None:
            # try:
            #     assert embedDim != -1
            # except AssertionError as e:
            #     raise (AssertionError("please specify embedding dimension! %s" % e))
            #initialise embedding
            vocab_size = len(self.vocab)
            #sd = 1 / np.sqrt(embedDim)  # Standard deviation to use

            #determine embedDim
            with open(embedPath, encoding="utf-8", mode="r") as textFile:
                embedDim = len(textFile.readline().split()) -1  #-1 to exclude actual word

            sd = 0.01
            self.embeddings = np.random.normal(0, scale=sd, size=[vocab_size, embedDim])
            self.embeddings = self.embeddings.astype(np.float32)

            with open(embedPath, encoding="utf-8", mode="r") as textFile:
                for line in textFile:
                    # Separate the values from the word
                    line = line.split()
                    word = line[0]

                    # If word is in our vocab, then update the corresponding weights
                    id = self.vocab.get(word, None)
                    if id is not None:
                        self.embeddings[id] = np.array(line[1:], dtype=np.float32)


        self.unkInd = self.vocab[self.datasetParams.unk_word]
        self.padInd = self.vocab[self.datasetParams.pad_word]

        tagsPath = os.path.join(path, 'tags.txt')
        self.tagMap = {}
        self.idxToTag = {}
        with open(tagsPath) as f:
            for idx, tag in enumerate(f.read().splitlines()):
                self.tagMap[tag] = idx
                self.idxToTag[idx] = tag

        noOfTags = len(self.tagMap)



        self.datasetParams.tagMap = self.tagMap
        self.datasetParams.padInd = self.padInd
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
        wordIdcs = []
        wordTokenized = nltk.word_tokenize(sentence)
        s = [self.vocab[token] if token in self.vocab
             else self.unkInd
             for token in wordTokenized]
        wordIdcs.append(s)
        sentenceChars = []
        for word in wordTokenized:
            wordChars = []
            for c in word:
                if c in self.charMap:
                    wordChars.append(self.charMap[c])
                else:
                    wordChars.append(self.unkInd)
            sentenceChars.append(wordChars)
        word_lengths = [len(word) for word in sentenceChars]
        max_word_length = max(word_lengths)
        chars = [word + [0] * (max_word_length - len(word)) for word in sentenceChars]
        return wordTokenized, wordIdcs, chars


    def loadSentencesLabels(self, sentencesFile, labelsFile):
        """ Loads sentences and labels , maps tokens and tags to their indices

        :param sentencesFile: file containing sentences
        :param labelsFile: file containing labels

        :return: dictionary containing loaded data
        """
        sentences = []
        allChars = []
        labels = []
        data = {}


        with open(sentencesFile, encoding=self.encoding) as f:
            for sentence in f.read().splitlines():
                # replace each token by its index if it is in vocab
                # else use index of unknown word replacement
                # s = [self.vocab[token] if token in self.vocab
                #      else self.unkInd
                #      for token in sentence.split(' ')]

                sentenceChars = []
                s = []
                for token in sentence.split(' '):
                    w = self.vocab[token] if token in self.vocab else self.unkInd
                    s.append(w)
                    if self.loadCharEmbed:
                        wordChars = []
                        for c in token:
                            if c in self.charMap:
                                wordChars.append(self.charMap[c])
                            else:
                                wordChars.append(self.unkInd)
                        sentenceChars.append(wordChars)
                allChars.append(sentenceChars)
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

        data['characters'] = allChars
        data['sentences'] = sentences
        data['labels'] = labels
        data['size'] = len(sentences)

        #data
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
                data[split] = self.loadSentencesLabels(sentencesFile, labelsFile)
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
            if self.loadCharEmbed:
                batchChars = [data['characters'][idx] for idx in order[batch * params.batch_size:(batch + 1) * params.batch_size]]
            batchSentences = [data['sentences'][idx] for idx in order[batch * params.batch_size:(batch + 1) * params.batch_size]]
            # batchSentences.sort(key=len, reverse=True)
            # batchSentences, batchTags =
            # print(type(batchSentences))
            batchTags = [data['labels'][idx] for idx in order[batch * params.batch_size:(batch + 1) * params.batch_size]]
            longestSentence = max([len(s) for s in batchSentences])

            # longestWord = 0
            # for sentence in batchChars:
            #     longest = max([len(w) for w in sentence])
            #     if longest > longestWord:
            #         longestWord = longest

            if self.loadCharEmbed:
                seq_lengths_in_words = np.array([len(i) for i in batchSentences])
                batch_maxlen = seq_lengths_in_words.max()
                char_batch = [sentence + [[0]] * (batch_maxlen - len(sentence)) for sentence in batchChars]
                word_lengths = [len(word) for sentence in char_batch for word in sentence]
                max_word_length = max(word_lengths)
                chars = [word + [0] * (max_word_length - len(word)) for sentence in char_batch for word in sentence]
                # print(chars)
                chars = torch.LongTensor(chars)
                # print(chars)
            else:
                chars = None


            # charsBatch = []
            # for sentence in char_batch:
            #     charsSentence = []
            #     for word in sentence:
            #         charsSentence.append(word + [0] * (max_word_length - len(word)))
            #     charsBatch.append(charsSentence)
            #print(charsBatch)



            # prepare a numpy array with the data, initialising the data with pad_ind and all labels with -1
            # initialising labels to -1 differentiates tokens with tags from PADding tokens
            batchSentencesPadded = self.padInd * np.ones((len(batchSentences), longestSentence)) #
            batchLabelsPadded = -1 * np.ones((len(batchSentences), longestSentence)) #
            #
            # pad char idx mapping
            X_char = []
            #batchCharsPadded = []
            # batchCharsLength = []
            # dAll = []
            # for sentence in batchChars:
            #     chars2_sorted = sorted(sentence, key=lambda p: len(p), reverse=True)
            #     d = {}
            #     for i, ci in enumerate(sentence):
            #         for j, cj in enumerate(chars2_sorted):
            #             if ci == cj and not j in d and not i in d.values():
            #                 d[j] = i
            #                 continue
            #     while len(chars2_sorted) < longestSentence:
            #         chars2_sorted.append([])
            #     dAll.append(d)
            #     chars2_length = [len(c) for c in chars2_sorted] #wordlength of words in sentence
            #     batchCharsLength.append(chars2_length)
            #     char_maxl = max(chars2_length)
            #     chars2_mask = np.zeros((len(chars2_sorted), longestWord), dtype='int')
            #     for i, c in enumerate(chars2_sorted):
            #         chars2_mask[i, :chars2_length[i]] = c
            #
            #     #print(chars2_mask)
            #     #chars2_mask = Variable(torch.LongTensor(chars2_mask))
            #     batchCharsPadded.append(chars2_mask)


            for j in range(len(batchSentences)):
                sentenceLen = len(batchSentences[j])
                batchSentencesPadded[j][:sentenceLen] = batchSentences[j]
                batchLabelsPadded[j][:sentenceLen] = batchTags[j]


            batchSentencesPadded = torch.LongTensor(batchSentencesPadded)
            batchLabelsPadded = torch.LongTensor(batchLabelsPadded)
            batchCharsPadded = torch.LongTensor(chars) if self.loadCharEmbed else None

            # shift tensors to GPU if available
            if torch.cuda.is_available():
                batchSentencesPadded = batchSentencesPadded.cuda()
                batchLabelsPadded = batchLabelsPadded.cuda()
                if self.loadCharEmbed:
                    batchCharsPadded = batchCharsPadded.cuda()

            # convert them to Variables to record operations in the computational graph
            batchSentencesPadded = Variable(batchSentencesPadded)
            batchLabelsPadded = Variable(batchLabelsPadded)
            if self.loadCharEmbed:
                batchCharsPadded = Variable(batchCharsPadded)

            yield batchSentencesPadded, batchLabelsPadded, batchCharsPadded