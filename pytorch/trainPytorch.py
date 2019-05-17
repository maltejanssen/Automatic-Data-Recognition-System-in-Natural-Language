import os
import tqdm
import logging
import torch.optim as optim
import torch
import numpy as np
from util import RunningAverage, loadCheckpoint, saveCheckpoint, configurateLogger, Params
from evaluate import evaluate
from DataLoader import DataLoader
import model.net
from ..commonUtil import *



def train(model, optimizer, lossFn, dataGenerator, metrics, params, numOfBatches):
    """ train model on numOfBatches batches

    :param torch.nn.Module model: model to be trained
    :param torch.optim optimizer: optimiser for parameters
    :param lossFn: loss function
    :param generator dataGenerator: generates batches of sentences and labels
    :param dict metrics: metrics to be applied to model
    :param Params params: hyperparameters
    :param int numOfBatches: number of batches to train on
    """
    model.train() #training mode

    summary = []
    lossAvg = RunningAverage()

    progressBar = tqdm.trange(numOfBatches)
    for batch in progressBar:
        trainBatch, labelsBatch = next(dataGenerator)


        outputBatch = model(trainBatch)
        loss = lossFn(outputBatch, labelsBatch)

        # clear previous gradients, compute gradients of all variables wrt loss
        optimizer.zero_grad()
        loss.backward()

        # update with calculated gradients
        optimizer.step()

        if batch % params.save_summary_steps == 0:
            outputBatch = outputBatch.data.cpu().numpy()
            labelsBatch = labelsBatch.data.cpu().numpy()

            batchSummary = {metric: metrics[metric](outputBatch, labelsBatch)
                             for metric in metrics}
            batchSummary['loss'] = loss.item() #.data[0]
            summary.append(batchSummary)

        lossAvg.update(loss.item()) #.data[0]
        progressBar.set_postfix(loss='{:05.3f}'.format(lossAvg()))


    metricsMean = {metric: np.mean([x[metric] for x in summary]) for metric in summary[0]}
    metricsString = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metricsMean.items())
    logging.info("- Train metrics: " + metricsString)



def train_and_evaluate(model, trainData, valData, optimiser, lossFn, metrics, params, modelDir, restaurationFile=None):
    """ Training and evaluation of epochs

    :param torch.nn.Module model: neural network
    :param dict trainData: training data
    :param dict valData: validation data
    :param torch.optim optimiser: optimiser for parameters
    :param lossFn: loss function
    :param dict metrics: metrics to be applied to model
    :param Params params: hyperparameters
    :param string modelDir: directory of nn model (containing config, weights and log)
    :param string restaurationFile: name of file to restore from (without extension)
    """

    if restaurationFile is not None:
        restorePath = os.path.join("model", restaurationFile + '.pth.tar')
        logging.info("Restoring parameters from {}".format(restorePath))
        loadCheckpoint(restorePath, model, optimiser)

    bestValAcc = 0.0

    paramsDir = r"experiments/base_model"

    for epoch in range(params.num_epochs):
        logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))

        #train epoch
        numOfBatches = (params.train_size + 1) // params.batch_size
        trainDataGenerator = dataLoader.batchGenerator(trainData, params, shuffle=True)
        train(model, optimiser, lossFn, trainDataGenerator, metrics, params, numOfBatches)

        #validate epoch
        numOfBatches = (params.val_size + 1) // params.batch_size
        valDataIterator = dataLoader.batchGenerator(valData, params, shuffle=False)
        valMetrics = evaluate(model, lossFn, valDataIterator, metrics, params, numOfBatches)

        valAcc = valMetrics['f1']
        isBest = valAcc >= bestValAcc

        # Save weights
        saveCheckpoint({'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'optim_dict': optimiser.state_dict()},
                        IsBest=isBest,
                        path=paramsDir)
        # modelPath = os.path.join(modelDir, "model.pt")
        # print(modelPath)
        # torch.save(model.state_dict(), modelPath)
        if isBest:
            logging.info("- Found new best accuracy")
            bestValAcc = valAcc

            bestJason = os.path.join(paramsDir, "metrics_val_best_weights.json")
            saveDict(valMetrics, bestJason)

        latestJason = os.path.join(paramsDir, "metrics_val_last_weights.json")
        saveDict(valMetrics, latestJason)


if __name__ == '__main__':
    paramsDir = r"experiments/base_model"

    # Load the parameters from json file
    jsonPath = os.path.join(paramsDir, "params.json")
    assert os.path.isfile(jsonPath), "No json configuration file found at {}".format(jsonPath)
    params = Params(jsonPath)

    # use GPU if available
    params.cuda = torch.cuda.is_available()

    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    if params.cuda:
        torch.cuda.manual_seed(230)

    # Set the logger
    configurateLogger(os.path.join(paramsDir, 'train.log'))

    # Create the input data pipeline
    logging.info("Loading the datasets...")

    # load data
    dataPath = "Data"
    encoding = "utf-8"
    dataLoader = DataLoader(dataPath, params, encoding)
    data = dataLoader.readData(dataPath, ["train", "val"])
    trainData = data["train"]
    #print(trainData)
    validationData = data["val"]

    params.train_size = trainData["size"]
    params.val_size = validationData["size"]

    logging.info("- done.")

    # Define the model and optimizer
    model = model.net.Net(params).cuda() if params.cuda else model.net.Net(params)
    optimiser = optim.Adam(model.parameters(), lr=params.learning_rate) #try out different optimisers!!

    netlossFn = model.lossFn
    netMetrics = model.metrics

    # Train the model
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    train_and_evaluate(model, trainData, validationData, optimiser, netlossFn, netMetrics, params, "model")

