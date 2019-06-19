import os
import torch
import numpy as np
from nltk import sent_tokenize
from DataLoader import DataLoader
from model.net2 import Net
from util import Params, prepareLabels
from torch.autograd import Variable


def translateIdcToLabel(lookup, predictions):
    translation = []
    for label in predictions:
        translation.append(lookup[label])
    return translation


def loadData(modelParamsFolder="experiments/base_model"):
    jsonPath = os.path.join(modelParamsFolder, "params.json")
    assert os.path.isfile(jsonPath), "No json configuration file found at {}".format(jsonPath)
    params = Params(jsonPath)

    dataPath = "Data"
    dataLoader = DataLoader(dataPath, params, encoding="utf-8")
    params.alphabet_size = dataLoader.datasetParams.alphabet_size
    params.tagMap = dataLoader.datasetParams.tagMap
    params.padInd = dataLoader.datasetParams.padInd
    return dataLoader, params


def loadModel(dataLoader, params):
    print(dataLoader.embeddings)
    model = Net(params, dataLoader.embeddings)
    state = torch.load("experiments/base_model/best.pth.tar")
    model.load_state_dict(state["state_dict"])
    model.eval()



def align_data(data):
    """Given dict with lists, creates aligned strings
    Adapted from Assignment 3 of CS224N
    Args:
        data: (dict) data["x"] = ["I", "love", "you"]
              (dict) data["y"] = ["O", "O", "O"]
    Returns:
        data_aligned: (dict) data_align["x"] = "I love you"
                           data_align["y"] = "O O    O  "
    """
    spacings = [max([len(seq[i]) for seq in data.values()])
                for i in range(len(data[list(data.keys())[0]]))]
    data_aligned = dict()

    # for each entry, create aligned string
    for key, seq in data.items():
        str_aligned = ""
        for token, spacing in zip(seq, spacings):
            str_aligned += token + " " * (spacing - len(token) + 1)

        data_aligned[key] = str_aligned

    return data_aligned


def get_model_api():
    dataLoader, params = loadData()
    print(dataLoader.embeddings)
    model = Net(params, dataLoader.embeddings)
    state = torch.load("experiments/base_model/best.pth.tar")
    model.load_state_dict(state["state_dict"])
    model.eval()

    def predict(text):
        words, wordIdcs, chars = dataLoader.loadSentences(text)
        wordIdcs = Variable(torch.LongTensor(wordIdcs))
        chars = Variable(torch.LongTensor(chars))
        output = model(wordIdcs, chars)

        crf = True if model.useCrf else False
        predictions, _ = prepareLabels(output, crf=crf)
        predictions = translateIdcToLabel(dataLoader.getidxToTag(), predictions)

        data = dict()
        data["words"] = words
        data["predictions"] = predictions
        print(predictions)
        print(words)
        alignedData = align_data(data)
        return alignedData

    return predict

