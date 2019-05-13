import Util


if __name__ == '__main__':
    tagsTrain = Util.readTags(r"Data\wnut\wnut17train.conll")
    tagsTest = Util.readTags(r"Data\wnut\emerging.test.conll")

    wordTaggedSentencesTrain, entitiesTrain = Util.tokenize(tagsTrain)
    wordTaggedSentencesTest, entitiesTest = Util.tokenize(tagsTest)

    posTaggedSentencesTrain = Util.posTag(wordTaggedSentencesTrain)
    posTaggedSentencesTest = Util.posTag(wordTaggedSentencesTest)

    completeTaggedSentencesTrain = Util.addEntitiyTaggs(posTaggedSentencesTrain, entitiesTrain)
    completeTaggedSentencesTest = Util.addEntitiyTaggs(posTaggedSentencesTest, entitiesTest)

    Util.writeToFile(completeTaggedSentencesTrain, r"Data\Corpus\wnut\train\train.conll")
    Util.writeToFile(completeTaggedSentencesTest, r"Data\Cotpus\wnut\test\test.conll")
    