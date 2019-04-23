from sklearn.feature_extraction import DictVectorizer
from Util import buildChunkTree
from train import chunkTrees2trainChunks
from nltk import tree2conlltags
from sklearn import ensemble
from Classifier import prev_next_pos_iob


def convertIntoSklearnFormat(parsedSentences, featureDetector):
    """ Transform a list of tagged sentences into a scikit-learn compatible format
       :param parsedSentences:
       :param featureDetector:
       :return:
       """
    X, y = [], []
    for parsed in parsedSentences:
        iobTagged = tree2conlltags(parsed)
        words, tags, iobTags = zip(*iobTagged))

        tagged = list(zip(words, tags))

        for index in range(len(iobTagged)):
            X.append(featureDetector(tagged, index, history=iobTags[:index]))
            y.append(iobTags[index])

    return X, y




trainChunkTrees = buildChunkTree(r"Data\Corpus" + "\\train")
# chunk trees have to be converted into trainable format| -> difference to Util.addEntitiyTaggs !!
#trainchunks = chunkTrees2trainChunks(trainChunkTrees)

evalChunkTrees = buildChunkTree(r"Data\Corpus" + "\\test")
#testChunks = chunkTrees2trainChunks(evalChunkTrees) ##???

X, y = convertIntoSklearnFormat(trainChunkTrees, prev_next_pos_iob)
print(X[0])
print(y[0])


# Xtrain, ytrain, XTest, yTest = convertIntoSklearnFormat(trainchunks, testChunks)
# clf = ensemble.RandomForestClassifier()
# clf.fit(Xtrain, ytrain)
#
# accuracy = clf.score(XTest, yTest)
# print("RandFor accuracy:", (accuracy) * 100)