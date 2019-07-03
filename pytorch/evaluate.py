import numpy as np
import torch
from utils.util import Params, prepareLabels


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
    return metricsMean
