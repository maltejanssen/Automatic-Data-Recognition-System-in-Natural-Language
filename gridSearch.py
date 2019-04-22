from sklearn.feature_extraction import DictVectorizer
from Util import buildChunkTree
from train import chunkTrees2trainChunks
from sklearn import ensemble


def convertIntoSklearnFormat(data):
    vectorizer = DictVectorizer()

    X = []
    y = []

    for sentence in data:
        for word in sentence:
            print(word)
            wordPos = word [0]
            entity = word[1]
            X.append(wordPos)
            y.append(entity)

    return X,y



trainChunkTrees = buildChunkTree(r"Data\Corpus" + "\\train")
# chunk trees have to be converted into trainable format| -> difference to Util.addEntitiyTaggs !!
trainchunks = chunkTrees2trainChunks(trainChunkTrees)

evalChunkTrees = buildChunkTree(r"Data\Corpus" + "\\test")
testChunks = chunkTrees2trainChunks(evalChunkTrees) ##???

X, y = convertIntoSklearnFormat(trainchunks)
print(X)
print(y)


# Xtrain, ytrain, XTest, yTest = convertIntoSklearnFormat(trainchunks, testChunks)
# clf = ensemble.RandomForestClassifier()
# clf.fit(Xtrain, ytrain)
#
# accuracy = clf.score(XTest, yTest)
# print("RandFor accuracy:", (accuracy) * 100)