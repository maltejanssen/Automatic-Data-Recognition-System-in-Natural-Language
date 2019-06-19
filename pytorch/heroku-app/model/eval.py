import numpy as np
from seqeval.metrics import f1_score


def filterPadding(goldL, pred):
    mask = (goldL >= 0)
    goldL = goldL[mask]
    pred = pred[mask]
    return goldL, pred

def accuracy(predictions, goldLabels, crf=None):
    """ calculates accuracy, excludes paddings

    :param np.ndarray outputLabels: log softmax output of the model
    :param np.ndarray goldLabels: indices of labels (-1 if padding) [0, 1, ... num_tag-1]
    :return float accuracy: in interval [0,1]
    """
    # #old -> doesnt work for f1
    # # flat vector conversion
    # goldLabels = goldLabels.ravel()
    # # mask to exclude padding tokens
    # mask = (goldLabels >= 0)
    # predictions = np.argmax(outputLabels, axis=1)
    # accuracy = np.sum((predictions == goldLabels) / float(np.sum(mask)))

    # flat vector conversion
    # goldLabels = goldLabels.ravel()
    # if not crf:
    #     predictions = np.argmax(outputLabels, axis=1)
    # else:
    #     predictions = outputLabels
    # #get all indices with pad token; cant delete in for loop
    # idcs = []
    # for idx, label in enumerate(goldLabels):
    #     if label == -1:
    #         idcs.append(idx)
    #
    # goldLabels = np.delete(goldLabels, idcs)
    # predictions = np.delete(predictions, idcs)
    # #goldLabels, predictions = filterPadding(goldLabels, predictions)

    accuracy = np.sum((predictions == goldLabels) / len(predictions))

    return accuracy
#TODO define more functions such as f1 and accuracy for each label


def f1(predictions, goldLabels, crf=None):


    # goldLabels = goldLabels.ravel()
    #
    # if not crf:
    #     predictions = np.argmax(outputLabels, axis=1)
    #     idcs = []
    #     for idx, label in enumerate(goldLabels):
    #         if label == -1:
    #             idcs.append(idx)
    #     goldLabels = np.delete(goldLabels, idcs)
    #     predictions = np.delete(predictions, idcs)
    # else:
    #     #
    #
    # #delete padding tokens from goldLabels
    # idcs = []
    # for idx, label in enumerate(goldLabels):
    #     if label == -1:
    #         idcs.append(idx)
    # goldLabels = np.delete(goldLabels, idcs)
    #
    # if not crf:
    #     predictions = np.argmax(outputLabels, axis=1)
    #     predictions = np.delete(predictions, idcs)
    # #goldLabels, predictions = filterPadding(goldLabels, predictions)

    tagsPath = "Data/tags.txt"

    #DataLoader ?? maybe cereate simplified version of Dataloader
    idxToTag = {}
    with open(tagsPath) as f:
        for idx, tag in enumerate(f.read().splitlines()):
            idxToTag[idx] = tag

    assert len(goldLabels) == len(predictions)
    goldLabelsTranslation = []
    predictionsTranslation = []
    for gold, pred in zip(goldLabels, predictions):
        goldLabelsTranslation.append(idxToTag[gold])
        predictionsTranslation.append(idxToTag[pred])

    f1Score = f1_score(goldLabelsTranslation, predictionsTranslation)
    #print(f1Score)
    return f1Score