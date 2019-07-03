import os
import torch
import json
import shutil
import numpy as np
from itertools import chain


def loadCheckpoint(checkpoint, model, optimizer=None):
    """ loads dictionary describing State of model from file

    :param string checkpoint: filename of parameters
    :param torch.nn.Module model: model of neural net
    :param torch.optim optimizer:
    :return: model
    """
    if not os.path.exists(checkpoint):
        raise ("File doesn't exist {}".format(checkpoint))
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint['state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    return checkpoint


def saveCheckpoint(state, IsBest, path):
    """ Saves models state at checkpoint

    :param dict state: describes models state
    :param bool IsBest: best model seen until now?
    :param str path: safe location
    """
    filepath = os.path.join(path, "last.pth.tar")
    if not os.path.exists(path):
        print("Checkpoint Directory does not exist! Making directory {}".format(path))
        os.mkdir(path)
    torch.save(state, filepath)
    if IsBest:
        shutil.copyfile(filepath, os.path.join(path, "best.pth.tar"))


class RunningAverage():
    """ class for calculation running average
    """

    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def __call__(self):
        return self.total / float(self.steps)


class Params():
    """ CLass that loads and saves models hyperparameters in json file

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
        """saves hyperparameters to json file

        :param jsonPath: save path
        """
        with open(jsonPath, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, jsonPath):
        """Loads parameters from json file"""
        with open(jsonPath) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """dict-like access to Params instance (`params.dict['learning_rate'])"""
        return self.__dict__


def writeResultToFile(words, tags, filename="results.txt"):
    """ writes predicted results for Sentence to File

    :param list(str) words: words of sentence
    :param list(str) tags: tags belonging to words
    :param filename: name of file that is to be written
    """
    path = "results"
    if not os.path.exists(path):
        os.mkdir(path)
    path = os.path.join(path, filename)

    with open(path, "a", encoding='utf-8') as fp:
        for word, tag in zip(words, tags):
            fp.write(word + "\t" + tag + "\n")
        fp.write("\n")


def saveDict(dictionary, path):
    """ Safes dictionary to json file

    :param dict dictionary: dictionary of float castable values
    :param path: Safe path of json file
    """
    with open(path, 'w') as f:
        # json needs float values
        dictionary = {k: float(v) for k, v in dictionary.items()}
        json.dump(dictionary, f, indent=4)



def prepareLabels(outputBatch, goldLabels=None, crf=False):
    """ translates output of net
    if goldLabels given prpares goldLabels and output for metrics calculation
    else: translates output of net into list of label indices

    :param outputBatch: net output
    :param goldLabels:
    :param bool crf: if True dataset format from crf output, false: softmax output
    :return predictions: flat list of predicted Labels
    :return np.array goldLabels: flat array of gold Labels
    """
    if goldLabels is not None:
        goldLabels = goldLabels.data.cpu().numpy()
        goldLabels = goldLabels.ravel()
        idcs = []
        for idx, label in enumerate(goldLabels):
            if label == -1:
                idcs.append(idx)
        goldLabels = np.delete(goldLabels, idcs)
    else:
        goldLabels = None

    if not crf:
        outputBatch = outputBatch.data.cpu().numpy()
        predictions = np.argmax(outputBatch, axis=1)
        if goldLabels is not None:
            predictions = np.delete(predictions, idcs)
        predictions = predictions.tolist()
    else:
        # convert from crf output format to flat list of batches tag
        outputBatch = np.array(outputBatch)
        allPreds = []
        for sentence in outputBatch:
            allPreds.append(sentence)
        predictions = list(chain.from_iterable(allPreds))

    return predictions, goldLabels


def translateIdcToLabel(lookup, predictions):
    translation = []
    for label in predictions:
        translation.append(lookup[label])
    return translation



