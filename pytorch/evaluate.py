import argparse
import sys
import logging
import os
import numpy as np
import torch
from util import Params, configurateLogger, loadCheckpoint, saveDict, prepareLabels
import model.net
from DataLoader import DataLoader
#from ..commonUtil import *

def evaluate(model, dataGenerator, metrics, params, numOfBatches):
    """ Evaluate model on NumOfBatches

    :param torch.nn.Module model: model to be trained
    :param lossFn: loss function
    :param generator dataGenerator: generates batches of sentences and labels
    metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
    :param Params params: hyperparameters
    :param int numOfBatches: number of batches to train on
    :return dict metricsMean: mean of different metrics
    """
    model.eval()
    summary = []

    for batch in range(numOfBatches):
        sentsBatch, labelsBatch, charsBatch = next(dataGenerator)
        #outputBatch = model(sentsBatch, charsBatch, labelsBatch)
        outputBatch = model(sentsBatch, charsBatch)

        try:
            params.crf
        except AttributeError as e:
            loss = model.loss(outputBatch, labelsBatch)
            crf = False
        else:
            #create non pad mask
            mask = torch.autograd.Variable(sentsBatch.data.ne(params.padInd)).float()
            crf = True
            loss = model.loss(outputBatch, labelsBatch, mask)

        if crf:
            outputBatch = model.crflayer.viterbi_decode(outputBatch, mask)

        #calculate metrics
        predictions, goldLabels = prepareLabels(outputBatch, labelsBatch, crf)
        batchSummary = {metric: metrics[metric](predictions, goldLabels, crf)
                        for metric in metrics}
        batchSummary['loss'] = loss.item()
        summary.append(batchSummary)

    metricsMean = {metric:np.mean([x[metric] for x in summary]) for metric in summary[0]}
    metricsString = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metricsMean.items())
    logging.info("- Eval metrics : " + metricsString)
    return metricsMean


if __name__ == '__main__':
    restoreFileStr = None # before executing set resore file!!!
    modelParamsFolder = "experiments/base_model"
    jsonPath = os.path.join(modelParamsFolder, "params.json")
    assert os.path.isfile(jsonPath), "No json configuration file found at {}".format(jsonPath)
    params = Params(jsonPath)

    params.cuda = torch.cuda.is_available()  # use GPU is available
    torch.manual_seed(230)
    if params.cuda:
        torch.cuda.manual_seed(230)

    configurateLogger(os.path.join(modelParamsFolder, 'evaluate.log'))
    logging.info("Creating the dataset...")

    encoding = "utf-8"
    dataPath = "Data"

    dataLoader = DataLoader(dataPath, params, encoding)
    data = dataLoader.readData(dataPath, ["test"])
    testData = data["test"]

    params.test_size = testData["size"]
    testDataGenerator = dataLoader.batchGenerator(testData, params)
    logging.info("- done.")

    model = model.net2.Net(params, dataLoader.embeddings).cuda() if params.cuda else model.net.Net(params, dataLoader.embeddings)


    metrics = model.metrics
    
    logging.info("Starting evaluation")
    restoreFileStr = "best"
    assert restoreFileStr != None, "no resote file set"
    restoreFile = os.path.join(modelParamsFolder, restoreFileStr + '.pth.tar')
    assert os.path.isfile(jsonPath), "No restore file found at {}".format(restoreFile)
    loadCheckpoint(restoreFile, model)

    numOfBatches = (params.test_size + 1) // params.batch_size
    result = evaluate(model, testDataGenerator, metrics, params, numOfBatches)
    savePath = os.path.join(modelParamsFolder, "metrics_test_{}.json".format(restoreFileStr))
    saveDict(result, savePath)
