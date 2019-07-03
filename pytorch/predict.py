import os
import torch
import argparse
from utils import util
from dataLoader.DataLoader import DataLoader
from torch.autograd import Variable
import numpy as np
from model.net2 import Net


parser = argparse.ArgumentParser()
parser.add_argument("--paramsDir",
                    help="Directory containing params.json")
parser.add_argument("--eval", default=False, action='store_true',
                    help='do evaluation')
parser.add_argument("--text", help="text that is to be classified")

path = os.path.join("results", "prediction")
if not os.path.exists(path):
    os.mkdir(path)
path = os.path.join(path, "results.txt")
if os.path.exists(path):
    os.remove(path)


def predictTestData(params, dataLoader, model):
    """ loads test data and predicts all entites + saves them in file

    :param Params params: params of the model
    :param DataLaoder dataLoader: dataloader
    :param model: model to evaluate dataset
    """
    dataPath = "Data"
    testData = dataLoader.readData(dataPath, ["test"])
    testData = testData["test"]
    params.test_size = testData["size"]
    testDataGenerator = dataLoader.batchGenerator(testData, params)
    numOfBatches = (params.test_size + 1) // params.batch_size


    for batch in range(numOfBatches):
        sentsBatch, labelsBatch, charsBatch = next(testDataGenerator)
        output = model(sentsBatch, charsBatch)
        crf = True if model.useCrf else False
        if crf:
            mask = torch.autograd.Variable(sentsBatch.data.ne(params.padInd)).float()
            output = model.crflayer.viterbi_decode(output, mask)

        prediction, gold = util.prepareLabels(output, labelsBatch, crf=crf)
        prediction = util.translateIdcToLabel(dataLoader.getidxToTag(), prediction)
        gold = util.translateIdcToLabel(dataLoader.getidxToTag(), gold)

        writeResultstoFile(prediction, gold, sentsBatch, dataLoader.idxToWord, dataLoader.padInd)


def writeResultstoFile(predictions, gold, sentences, idxToWord, padInD):
    """ writes prediction results to file in wnutEvalFormat: line: word gold prediction

    :param list predictions: list of predictions
    :param gold:  list of gold labels
    :param sentences: list of list of sentence index mapping
    :param idxToWord: idxToWord lookup
    :param padInD: Padding index
    """
    sentences = sentences.data.cpu().numpy()
    sentences = sentences.ravel()

    idcs = []
    for idx, label in enumerate(sentences):
        if label == padInD:
            idcs.append(idx)
    sentences = np.delete(sentences, idcs)


    with open(path, "a", encoding='utf-8') as fp:
        for idx, p in enumerate(sentences):
            fp.write("".join("{}\t{}\t{}".format(idxToWord[p], gold[idx], predictions[idx])))
            fp.write("\n")


def predict(dataLoader, text, model):
    """ predicts the given text

    :param DataLoader dataLoader: dataLoader
    :param str text: text to be classified
    :param Net model: mdoel to classify text
    :return: list of predictions
    """
    words, wordIdcs, chars = dataLoader.loadSentences(text)
    wordIdcs = torch.LongTensor(wordIdcs)
    chars = torch.LongTensor(chars)
    output = model(wordIdcs, chars)

    crf = True if model.useCrf else False
    if crf:
        mask = torch.autograd.Variable(wordIdcs.data.ne(params.padInd)).float()
        output = model.crflayer.viterbi_decode(output, mask)
    predictions, _ = util.prepareLabels(output, crf=crf)
    predictions = util.translateIdcToLabel(dataLoader.getidxToTag(), predictions)

    return predictions


if __name__ == '__main__':
    args = parser.parse_args()

    if args.paramsDir is None:
        paramsDir = os.path.join("experiments", "bi-LSTMpretrainedEmbedcharembdecrf")
    else: paramsDir = args.paramsDir

    jsonPath = os.path.join(paramsDir, "params.json")
    assert os.path.isfile(jsonPath), "No json configuration file found at {}".format(jsonPath)
    params = util.Params(jsonPath)

    dataPath = "Data"
    dataLoader = DataLoader(dataPath, params, encoding="utf-8")
    params.alphabet_size = dataLoader.datasetParams.alphabet_size
    params.tagMap = dataLoader.datasetParams.tagMap
    params.padInd = dataLoader.datasetParams.padInd


    model = Net(params, dataLoader.embeddings)
    state = torch.load(os.path.join(paramsDir, "best.pth.tar"))
    model.load_state_dict(state["state_dict"])
    model.eval()


    if args.eval:
        predictTestData(params, dataLoader, model)
    else:
        with torch.no_grad():
            assert args.text is not None
            prediction = predict(dataLoader, args.text, model)
            print("prediction:")
            print(prediction)