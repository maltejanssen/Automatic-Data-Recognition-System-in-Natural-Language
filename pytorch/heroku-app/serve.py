import os
import torch
from torch.autograd import Variable
from dataLoader.DataLoader import DataLoader
from model.net2 import Net
from utils.util import Params, prepareLabels



def translateIdcToLabel(lookup, predictions):
    translation = []
    for label in predictions:
        translation.append(lookup[label])
    return translation


def loadData(modelParamsFolder="experiments/bi-LSTMpretrainedEmbedcharembdecrf"):
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
    model = Net(params, dataLoader.embeddings)
    state = torch.load("experiments/bi-LSTMpretrainedEmbedcharembdecrf/best.pth.tar")
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


def turnIntoSpacyFormat(predictions, words, text):
    """turns given prediction into spacy fomat for visualisation

    :param list predictions: list of predections
    :param list words: list of words belonging to the prediction
    :param str text: text belonging to prediction
    :return: spacy formated dict defining positions of entities
    """
    print(predictions, flush=True)
    print(words, flush=True)
    print(text, flush=True)
    entities = []
    current = 0
    idx = 0
    while idx < len(predictions):
        if predictions[idx][0] != "O":
            if predictions[idx][0] == "B" or predictions[idx][0] == "I":
                d = {}
                d["start"] = current
                d["label"] = predictions[idx][2:]
                d["end"] = current + len(words[idx])
                current += len(words[idx])
                if current < len(text):
                    if text[current] == " ":
                        current += 1
                idx += 1
            try:
                next = predictions[idx]
            except:
                entities.append(d)
                break
            while next[0] == "I":
                d["end"] = current + len(words[idx])
                current += len(words[idx])
                try:
                    idx += 1
                    next = predictions[idx]
                except:
                    break
                if current < len(text):
                    if text[current] == " ":
                        current += 1
            entities.append(d)
        try:
            current += len(words[idx])
        except:
            break
        if current < len(text):
            if text[current] == " ":
                current += 1
        idx += 1
    return entities



def getModelApi():
    dataLoader, params = loadData()
    model = Net(params, dataLoader.embeddings)
    state = torch.load("experiments/bi-LSTMpretrainedEmbedcharembdecrf/best.pth.tar")
    model.load_state_dict(state["state_dict"])
    model.eval()

    def predict(text):
        words, wordIdcs, chars = dataLoader.loadSentences(text)
        wordIdcs = Variable(torch.LongTensor(wordIdcs))
        chars = Variable(torch.LongTensor(chars))
        output = model(wordIdcs, chars)

        crf = True if model.useCrf else False
        if crf:
            mask = torch.autograd.Variable(wordIdcs.data.ne(params.padInd)).float()
            output = model.crflayer.viterbi_decode(output, mask)
        predictions, _ = prepareLabels(output, crf=crf)
        predictions = translateIdcToLabel(dataLoader.getidxToTag(), predictions)
        data = dict()
        data["words"] = words
        data["predictions"] = predictions
        data["text"] = text
        alignedData = align_data(data)
        alignedData["text"] = text
        return data, alignedData, turnIntoSpacyFormat(predictions, words, text)

    return predict

