import os
import json


class DataLoader(object):
    """
    Handles all aspects of the data. Stores the dataset_params, vocabulary and tags with their mappings to indices.
    """

    def __init__(self, data_dir, params):
        """
        Loads dataset_params, vocabulary and tags. Ensure you have run `build_vocab.py` on data_dir before using this
        class.

        Args:
            data_dir: (string) directory containing the dataset
            params: (Params) hyperparameters of the training process. This function modifies params and appends
                    dataset_params (such as vocab size, num_of_tags etc.) to params.
        """

        # loading dataset_params
        jasonPath = "Data/dataset_params.json"
        assert os.path.isfile(jasonPath), "No json file found at {}, run build_vocab.py".format(jasonPath)
        self.datasetParams = Params(jasonPath)

        # loading vocab (we require this to map words to their indices)
        VocabPath = "Data/words.txt"
        self.vocab = {}
        with open(VocabPath) as f:
            for i, l in enumerate(f.read().splitlines()):
                self.vocab[l] = i

        # setting the indices for UNKnown words and PADding symbols
        self.unkInd = self.vocab[self.datasetParams.unk_word]
        self.padInd = self.vocab[self.datasetParams.pad_word]

        # loading tags (we require this to map tags to their indices)
        TagsPath = os.path.join(data_dir, 'tags.txt')
        self.tagMap = {}
        with open(TagsPath) as f:
            for i, t in enumerate(f.read().splitlines()):
                self.tagMap[t] = i

        # adding dataset parameters to param (e.g. vocab size, )
        params.update(jasonPath)


class Params():
    """Class that loads hyperparameters from a json file.

    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, jsonPath):
        with open(jsonPath) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, jsonPath):
        with open(jsonPath, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, jsonPath):
        """Loads parameters from json file"""
        with open(jsonPath) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__


def openVocab(path):
    vocab = {}
    with open(path) as f:
      for i, l in enumerate(f.read().splitlines()):
        vocab[l] = i


def loadData(vocabWords, vocabTags):
    train_sentences = []
    train_labels = []

    with open("Data/net/train") as f:
        for sentence in f.read().splitlines():
            # replace each token by its index if it is in vocab
            # else use index of UNK
            s = [vocabWords[token] if token in self.vocab
                 else vocabWords['UNK']
                 for token in sentence.split(' ')]
            train_sentences.append(s)

    with open("Data/net/train") as f:
        for sentence in f.read().splitlines():
            # replace each label by its index
            l = [vocabTags[label] for label in sentence.split(' ')]
            train_labels.append(l)


vocabWords = openVocab("Data/words.txt")
vocabTags = openVocab("Data/tags.txt")
