import Util


if __name__ == '__main__':
    tagsTrain = Util.readTags(r"Data\wnut\wnut17train.conll")
    tagsTest = Util.readTags(r"Data\wnut\emerging.test.annotated")
    tagsVal = Util.readTags(r"Data\wnut\emerging.dev.conll")

    wordTaggedSentencesTrain, entitiesTrain = Util.tokenize(tagsTrain)
    wordTaggedSentencesTest, entitiesTest = Util.tokenize(tagsTest)
    wordTaggedSentencesVal, entitiesVal = Util.tokenize(tagsVal)

    posTaggedSentencesTrain = Util.posTag(wordTaggedSentencesTrain)
    posTaggedSentencesTest = Util.posTag(wordTaggedSentencesTest)
    posTaggedSentencesVal = Util.posTag(wordTaggedSentencesVal)

    completeTaggedSentencesTrain = Util.addEntitiyTaggs(posTaggedSentencesTrain, entitiesTrain)
    completeTaggedSentencesTest = Util.addEntitiyTaggs(posTaggedSentencesTest, entitiesTest)
    completeTaggedSentencesVal = Util.addEntitiyTaggs(posTaggedSentencesVal, entitiesVal)

    Util.writeTripletsToFile(completeTaggedSentencesTrain, r"Data\Corpus\train\train.conll")
    Util.writeTripletsToFile(completeTaggedSentencesTest, r"Data\Corpus\test\test.conll")
    Util.writeTripletsToFile(completeTaggedSentencesVal, r"Data\Corpus\val\val.conll")
