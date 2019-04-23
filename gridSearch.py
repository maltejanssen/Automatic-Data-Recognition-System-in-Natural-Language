from Util import buildChunkTree
from features import prev_next_pos_iob
import nltk


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

trainChunkTrees = buildChunkTree(r"Data\Corpus" + "\\train")
evalChunkTrees = buildChunkTree(r"Data\Corpus" + "\\test")

X, y = convertIntoSklearnFormat(trainChunkTrees, prev_next_pos_iob)
print(X[0])
print(y[0])

# Xtrain, ytrain, XTest, yTest = convertIntoSklearnFormat(trainchunks, testChunks)
# clf = ensemble.RandomForestClassifier()
# clf.fit(Xtrain, ytrain)
#
# accuracy = clf.score(XTest, yTest)
# print("RandFor accuracy:", (accuracy) * 100)