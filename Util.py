from collections import namedtuple
import nltk


Tag = namedtuple("Tag", ["word", "tag"])

def readTags(file):
    """
    Creates a list of tagged words from the corpus

    Parameter
    String file - dest of file from which sentences are to be read

    Return
    sentences - read tags
    """
    tags = []
    sep = "\t"
    with open(file, encoding='utf-8') as fp:
        for line in fp:
            line = line.strip()
            if line:
                line = line.split(sep)
                tags.append(Tag(*line))
            else:
                tags.append(Tag("", ""))  # append emty tuple to mark sentence ending
    return tags


def tokenize(tags):
    """ sentence and word tokenization
    Parameter
    tags - List of tags to be tokenised

    Return

    """

    words = []
    entities = []
    sentence = []
    entitiesOfSentence = []

    for tag in tags:
        if (tag[0] == "" and tag[1] == ""):
            words.append(sentence)
            entities.append(entitiesOfSentence)
            sentence = []
            entitiesOfSentence = []
        else:
            sentence.append(tag[0])
            entitiesOfSentence.append(tag[1])

    return words, entities


def addEntitiyTaggs(posTagged, entities):
    """ adds Entities to posTagged list
    Parameter
    posTAgged - List of tags to be tokenised

    Return
    returns
    """

    if (len(posTagged) != len(entities)):
        raise ValueError

    newTags = []
    sentence = []
    i = 0
    for i in range(len(posTagged)):
        for j in range(len(posTagged[i])):
            sentence.append(((posTagged[i][j][0], posTagged[i][j][1]), entities[i][j]))
        newTags.append(sentence)
        sentence = []
    return newTags

def posTag(sentences):
    posTaggedSentences = [nltk.pos_tag(sent) for sent in sentences]
    return posTaggedSentences



if __name__ == '__main__':
    tagsTrain = readTags(r"Data\wnut17train.conll")
    print(tagsTrain[0:10])
    print(len(tagsTrain))

    tagsTest = readTags(r"Data\emerging.test.conll")  # error due to encoding
    print(tagsTest[0:10])

    wordTaggedSentencesTrain, entitiesTrain = tokenize(tagsTrain)
    wordTaggedSentencesTest, entitiesTest = tokenize(tagsTest)
    print(wordTaggedSentencesTrain)
    print(entitiesTrain)

    posTaggedSentencesTrain = posTag(wordTaggedSentencesTrain)
    posTaggedSentencesTest = posTag(wordTaggedSentencesTest)
    print(posTaggedSentencesTrain[0:10])

    completeTaggedSentencesTrain = addEntitiyTaggs(posTaggedSentencesTrain, entitiesTrain)
    completeTaggedSentencesTest = addEntitiyTaggs(posTaggedSentencesTest, entitiesTest)
    print(completeTaggedSentencesTrain[0:10])

