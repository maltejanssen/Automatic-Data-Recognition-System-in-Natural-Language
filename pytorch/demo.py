import os
import util
from util import loadCheckpoint
import model.net2
import torch
import util
import DataLoader
from torch.autograd import Variable
import numpy as np
from nltk import sent_tokenize,word_tokenize



def loadData(modelParamsFolder="experiments/base_model"):
    ### load params
    jsonPath = os.path.join(modelParamsFolder, "params.json")
    assert os.path.isfile(jsonPath), "No json configuration file found at {}".format(jsonPath)
    params = util.Params(jsonPath)

    dataPath = "Data"
    dataLoader = DataLoader.DataLoader(dataPath, params, encoding="utf-8")
    params.alphabet_size = dataLoader.datasetParams.alphabet_size
    params.tagMap = dataLoader.datasetParams.tagMap
    params.padInd = dataLoader.datasetParams.padInd
    return dataLoader, params


dataLoader, params = loadData()
print(dataLoader.embeddings)
model = model.net2.Net(params, dataLoader.embeddings)
state = torch.load("experiments/base_model/best.pth.tar")
model.load_state_dict(state["state_dict"])
model.eval()

sentences = "Peter Miller went to Madrid last weekend to go shopping at Walmart. Fuck the system"
def predict(text):
    sentTokensized = sent_tokenize(text)
    #sentenceList = [ for s in tokenized]
    allPredictions = []

    for sentence in sentTokensized:
        translation = []
        words, wordIdcs, chars = dataLoader.loadSentences(sentence)
        print(words)
        print(chars)
        print(wordIdcs)
        t = Variable(torch.LongTensor(wordIdcs))
        chars = Variable(torch.LongTensor(chars))
        output = model(t, chars)
        output = output.data.cpu().numpy()
        predictions = np.argmax(output, axis=1)

        lookup = dataLoader.getidxToTag()
        print(dataLoader.getidxToTag())
        for label in predictions:
            translation.append(lookup[label])
        allPredictions.append(translation)


        for word, tag in zip(words, translation):
            print(word, ':', tag)

        util.writeResultToFile(words, translation)

    return words, translation


predictions = predict(sentences)
