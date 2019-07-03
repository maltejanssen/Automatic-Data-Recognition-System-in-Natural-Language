import os
import json
import nltk
import numpy as np
from sklearn import ensemble, tree, svm, naive_bayes, linear_model
import sklearn.feature_extraction
from sklearn.metrics import make_scorer
from sklearn.preprocessing import LabelEncoder
from hypopt import GridSearch

from .dataLoader.dataLoader import buildChunkTree
from .model.features import extractFeatures



def saveDict(dictionary, path):
    """ Safes dictionary to json file

    :param dict dictionary: dictionary of float castable values
    :param path: Safe path of json file
    """
    with open(path, 'w') as f:
        # json needs float values
        dictionary = {k: float(v) for k, v in dictionary.items()}
        json.dump(dictionary, f, indent=4)



def convertIntoSklearnFormat(parsedSentences, featureDetector):
    """ Transform a list of tagged sentences into a scikit-learn compatible format
        detects features of parsedSentences and splits features from classes
       :param parsedSentences: list of lists containing IOB triplets as tuples for each word
       :type parsedSentences: List
       :param featureDetector: feature extraction function
       :type featureDetector: function
       :return X: list of extracted features of dataset
       :return y: classes belonging to X
       """
    X, y = [], []
    for parsed in parsedSentences:
        iobTagged = nltk.tree2conlltags(parsed)
        words, tags, iobTags = zip(*iobTagged)
        tagged = list(zip(words, tags))

        for index in range(len(iobTagged)):
            X.append(featureDetector(tagged, index, history=iobTags[:index]))
            y.append(iobTags[index])
    return X, y


def customf1(gold, pred):
    return sklearn.metrics.f1_score(gold, pred, labels=[0,1,2,3,4,5,6,7,8,9,10,11], average="micro")


def gridSearch(Xtrain, ytrain, Xval, yval, classifier, parameters):
    encoder = LabelEncoder()
    vectorizer = sklearn.feature_extraction.DictVectorizer(sparse=True)
    Xtrain = vectorizer.fit_transform(Xtrain)
    ytrain = encoder.fit_transform(ytrain)
    Xval = vectorizer.transform(Xval)
    yval = encoder.transform(yval)
    clf = GridSearch(model=classifier, param_grid=parameters)
    scorer = make_scorer(customf1)
    clf.fit(Xtrain, ytrain, Xval, yval, scoring = scorer)

    # clf = GridSearchCV(estimator=classifier, param_grid=parameters, scoring="f1")
    # clf.fit(Xtrain, ytrain)
    return clf.get_param_scores()


if __name__ == "__main__":
    #load files into nltk chunktrees
    trainChunkTrees = buildChunkTree(r"Data\Corpus" + "\\train")
    valChunkTrees = buildChunkTree(r"Data\Corpus" + "\\val")

    #convert into sklearn format
    Xtrain, ytrain = convertIntoSklearnFormat(trainChunkTrees, extractFeatures)
    Xval, yval = convertIntoSklearnFormat(valChunkTrees, extractFeatures)

    #define parameter grid
    parameters = [{'kernel': ('linear', "rbf", "poly", "sigmoid"), 'C': 10. ** np.arange(-3, 8),
                   "gamma": 10. ** np.arange(-5, 4)},
                  {'kernel': ('linear', "rbf", "poly", "sigmoid"), "nu": [0.2,0.4,0.6,0.8,1],
                   "gamma": 10. ** np.arange(-5, 4)},
                  {"penalty": ("l1", "l2"), "loss": ("squared_hinge", "hinge"), "dual": [True, False],
                   'C': 10. ** np.arange(-3, 8)},
                  {"criterion": ("gini", "entropy"), "max_depth": [3,5,10,20],
                   "min_samples_split": [1,2,3], "min_samples_leaf": [1,2,3]},
                  {'C': 10. ** np.arange(-3, 8,), "penalty":["l1","l2"]}]
    names = ["svm.txt", "nuSVC.txt", "linearSVC.txt", "decisionTree.txt", "logReg.txt"]
    models = [svm.SVC(), svm.NuSVC(), svm.LinearSVC(), tree.DecisionTreeClassifier(random_state=1),
              linear_model.LogisticRegression()]

    for idx, model in enumerate(models):
        result = gridSearch(Xtrain, ytrain, Xval, yval, models[idx], parameters[idx])
        resultDict = {}
        for entry in result:
            params, score = entry
            params = json.dumps(params)
            resultDict[params] = score
        path = os.path.join("results\gridtest", names[idx])
        saveDict(resultDict, path)


