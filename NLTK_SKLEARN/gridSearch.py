import os
from Util import buildChunkTree
from features import prev_next_pos_iob
import nltk
from sklearn import ensemble, tree, svm
import sklearn.feature_extraction
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
import pandas
from hypopt import GridSearch
from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score
from ..commonUtil import *


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


def gridSearch(Xtrain, ytrain, Xval, yval, classifier, parameters):
    encoder = LabelEncoder()
    vectorizer = sklearn.feature_extraction.DictVectorizer(sparse=True)
    Xtrain = vectorizer.fit_transform(Xtrain)
    ytrain = encoder.fit_transform(ytrain)
    Xval = vectorizer.transform(Xval)
    yval = encoder.transform(yval)
    clf = GridSearch(model=classifier, param_grid=parameters)
    clf.fit(Xtrain, ytrain, Xval, yval, scoring = "f1_micro")
    # clf = GridSearchCV(estimator=classifier, param_grid=parameters, scoring="f1")
    # clf.fit(Xtrain, ytrain)
    return clf.get_param_scores()


if __name__ == "__main__":
    #load files into nltk chunktrees
    trainChunkTrees = buildChunkTree(r"Data\Corpus" + "\\train")
    valChunkTrees = buildChunkTree(r"Data\Corpus" + "\\val")

    #convert into sklearn format
    Xtrain, ytrain = convertIntoSklearnFormat(trainChunkTrees, prev_next_pos_iob)
    Xval, yval = convertIntoSklearnFormat(valChunkTrees, prev_next_pos_iob)

    models = [svm.SVC(gamma="scale")]
    names = ["svm.txt"]
    parameters = [{'kernel':('linear', 'rbf'), 'C':[1, 10]}]

    for idx, model in enumerate(models):
        result = gridSearch(Xtrain, ytrain, Xval, yval, models[idx], parameters[idx])
        resultDict = {}
        for entry in result:
            params, score = entry
            resultDict[params] = score
        path = os.path.join(r"NLTK_SKLEARN\\results\\gridtest", names[idx])
        saveDict(resultDict, path)



#train sklearn classifier -> out of memory
# clf = tree.DecisionTreeClassifier(criterion="gini", max_features="auto", max_depth=100)
# vectorizer = sklearn.feature_extraction.DictVectorizer(sparse=False)
# vectorizer.fit(Xtrain)
# Xtrain = vectorizer.transform(X)
# clf.fit(Xtrain, ytrain)

#train sklearn classifier
# encoder = LabelEncoder()
# vectorizer = sklearn.feature_extraction.DictVectorizer(sparse=True)
# clf = tree.DecisionTreeClassifier(criterion="gini", max_features="auto", max_depth=100)
# Xtrain = vectorizer.fit_transform(Xtrain)
# ytrain = encoder.fit_transform(ytrain)
# clf.fit(Xtrain,ytrain)

#gridsearch
# encoder = LabelEncoder()
# vectorizer = sklearn.feature_extraction.DictVectorizer(sparse=True)
# svc = svm.SVC(gamma="scale")
# Xtrain = vectorizer.fit_transform(Xtrain)
# ytrain = encoder.fit_transform(ytrain)
#
# parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
# clf = GridSearchCV(svc, parameters, cv=5)
# clf.fit(Xtrain, ytrain)

