import os
from subprocess import check_call
import sys
import argparse

import util

PYTHON = sys.executable
parser = argparse.ArgumentParser()
parser.add_argument('--paramDir', default='experiments/learning_rate',
                    help='Directory containing params.json')
def lauchTrainingJob(paramDir, dataDir, hyperParamSearchDir, params):
    """Launch training of the model with a set of hyperparameters in parent_dir/job_name

    Args:
        saveDir: (string) directory containing config, weights and log
        dataDir: (string) directory containing the dataset
        params: (dict) containing hyperparameters
    """
    saveDir = os.path.join(paramDir, hyperParamSearchDir)
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)

    jsonPath = os.path.join(saveDir, 'params.json')
    params.save(jsonPath)

    cmd = "{python} trainPytorch.py --saveDir={saveDir}".format(python=PYTHON, paramsDir=saveDir)
    print(cmd)
    check_call(cmd, shell=True)


if __name__ == "__main__":
    args = parser.parse_args()
    # Load the "reference" parameters from parent_dir json file
    jsonPath = os.path.join(args.paramDir, 'params.json')
    assert os.path.isfile(jsonPath), "No json configuration file found at {}".format(jsonPath)
    params = util.Params(jsonPath)

    learningRates = [1e-4, 1e-3, 1e-2]

    for learning_rate in learningRates:
        params.learning_rate = learning_rate
        jobName = "learning_rate_{}".format(learning_rate)
        lauchTrainingJob(args.parent_dir, args.data_dir, jobName, params)

