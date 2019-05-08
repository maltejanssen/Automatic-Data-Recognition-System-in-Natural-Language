import os
import torch
import json
import shutil
import logging


def configurateLogger(path):
    """ configurates logger and safes terminal logging to file

    :param str path: safe path of logging file
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        fileHandler = logging.FileHandler(path)
        fileHandler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(fileHandler)

        streamHandler = logging.StreamHandler()
        streamHandler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(streamHandler)


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


def saveDict(dictionary, path):
    """ Safes dictionary to json file

    :param dict dictionary: dictionary of float castable values
    :param path: Safe path of json file
    """
    with open(path, 'w') as f:
        # json needs float values
        dictionary = {k: float(v) for k, v in dictionary.items()}
        json.dump(dictionary, f, indent=4)


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

