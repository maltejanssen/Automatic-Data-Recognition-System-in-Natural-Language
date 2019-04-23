from Util import buildChunkTree
from features import prev_next_pos_iob
import nltk
from sklearn import ensemble, tree, svm
import sklearn.feature_extraction
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder


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

#load files into nltk chunktrees
trainChunkTrees = buildChunkTree(r"Data\Corpus" + "\\train")
evalChunkTrees = buildChunkTree(r"Data\Corpus" + "\\test")

#convert into sklearn format
Xtrain, ytrain = convertIntoSklearnFormat(trainChunkTrees, prev_next_pos_iob)

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
encoder = LabelEncoder()
vectorizer = sklearn.feature_extraction.DictVectorizer(sparse=True)
svc = svm.SVC(gamma="scale")
Xtrain = vectorizer.fit_transform(Xtrain)
ytrain = encoder.fit_transform(ytrain)

parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
clf = GridSearchCV(svc, parameters, cv=5)
clf.fit(Xtrain, ytrain)



# vectorizer = sklearn.feature_extraction.DictVectorizer(sparse=False)
# vectorizer.fit(Xtrain)
# Xtrain = vectorizer.transform(X)
#
# parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
# svc = svm.SVC(gamma="scale")
# clf = GridSearchCV(svc, parameters, cv=5)
# clf.fit(Xtrain, ytrain)
