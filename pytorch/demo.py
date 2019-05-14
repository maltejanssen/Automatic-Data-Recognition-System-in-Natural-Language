import os
import util
from util import loadCheckpoint
import model.net
import torch
import util
import DataLoader
from torch.autograd import Variable
import numpy as np
from nltk import sent_tokenize


### load params
modelParamsFolder = "experiments/base_model"
jsonPath = os.path.join(modelParamsFolder, "params.json")
assert os.path.isfile(jsonPath), "No json configuration file found at {}".format(jsonPath)
params = util.Params(jsonPath)

dataPath = "Data"
dataLoader = DataLoader.DataLoader(dataPath, params)

#load model
model = model.net.Net(params)
model.load_state_dict(torch.load("experiments/base_model/model.pt"))
model.eval()


sentences = "Peter Miller went to Madrid last weekend to go shopping at Walmart. Fuck the system"


def predict(text):
    tokenized = sent_tokenize(text)
    #sentenceList = [ for s in tokenized]
    allPredictions = []

    for sentence in tokenized:
        translation = []
        words, wordIdcs = dataLoader.loadSentences(sentence)
        t = Variable(torch.LongTensor(wordIdcs))
        output = model(t)
        output = output.data.cpu().numpy()
        predictions = np.argmax(output, axis=1)

        lookup = dataLoader.getidxToTag()
        for label in predictions:
            translation.append(lookup[label])
        allPredictions.append(translation)


        for word, tag in zip(words, translation):
            print(word, ':', tag)

        util.writeResultToFile(words, translation)

    return words, translation


predictions = predict(sentences)
