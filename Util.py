from ReadWrite import readTags, writeToFile
import nltk
import sys, os
scriptDir = os.path.dirname(__file__)
path = os.path.join(scriptDir, "reader")
sys.path.insert(0, path)
from reader import ConllChunkCorpusReader


def tokenize(tags):
    """ performs word and sentence tokenizing

    :param tags: List of named tuples to be tokenised: ("Tag", ["word", "tag"])
    :return: words: list of lists which contain all words; Each list represents one sentence
    :return: entities: list of lists which contain all entities; Each list represents one sentence
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
    """ adds Entities to posTagged lists

    :param posTagged: list of lists containing part-of-speech tags for each word
    :param entities: entities corresponding to word in posTagged
    :return: newTags: list of lists of tuples containing word,POS-Tag,entity: (('word', 'POS-Tag'), 'entity')
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
    """ processes a sequence of words and adds part-of-speech Tag to each word

    :param sentences: list of lists, each list representing a sentence and contains it's words
    :return: posTaggedSentences: list of lists containing part-of-speech tags
            eg.: [[('@paulwalk', 'VB'), ('It', 'PRP'), ("'s", 'VBZ'), ('the', 'DT'), ('view', 'NN')...]...
            ('goal', 'NN'), ('for', 'IN'), ('today', 'NN'), ('!', '.')]]

    """
    posTaggedSentences = [nltk.pos_tag(sent) for sent in sentences]
    return posTaggedSentences



def buildChunkTree(corpusPath):
    """ reads a directory and converts its files into chunkTrees
    :param corpusPath: path of corpus(folder)(files to be converted into chunkTrees)
    :return: chunkTrees: chunked Sentences of read Data eg: [Tree('S', [('@paulwalk', 'VB'), ('It', 'PRP'), ("'s", 'VBZ'),...
    """
    reader = ConllChunkCorpusReader(corpusPath, ".*", ['person', 'location', 'corporation', 'product', 'creative-work', 'group'])
    chunkTrees = reader.chunked_sents()
    return chunkTrees


def getInstancesOfEntity(entity, data):
    """ returns all instances or entity x in Dataset

    :param entity: entity to search
    :param data: Tuples of iob Triblets: (word, pos), enityTag
    :return: List of all found entities
    """
    instances = []
    iobTriplets = []

    for sentences in data:
        sentence = []
        for triplet in sentences:
            sentence.append((triplet[0][0], triplet[0][1], triplet[1]))
        iobTriplets.append(sentence)
        sentence = []
    instances = []
    for sentence in iobTriplets:
        for triplet in sentence:
            if entity in triplet[2]:
                instances.append(triplet[0])
    return instances


def getAmountOfInstances(filepath):
    """counts amount of total words in file

    :param filepath: Path of file
    :return: amount of words
    """
    #len(data) not enough because of empty lines
    count = 0
    sep = "\t"
    with open(filepath, encoding='utf-8') as fp:
        for line in fp:
            line = line.strip()
            if line:
                count += 1
    return count


def calculatePercentageOfEntities(filepath):
    """calculates amount and percentage of each entity for statistical use

    :param filepath: path of file
    :return: dictionaries of amount and percentage of each entity in Dataset
    """
    #could be shorter if is reworked, but getInstancesOfEntity useful in other cases= new function or fine?
    tags = readTags(filepath)
    wordTaggedSentences, entities = tokenize(tags)
    posTaggedSentences = posTag(wordTaggedSentences)
    data = addEntitiyTaggs(posTaggedSentences, entities)

    AmountOfEntities = {"person":0, "location":0, "corporation":0, "product":0, "creative-work":0, "group":0, "O":0}
    percentageOfEntities = {"person":0, "location":0, "corporation":0, "product":0, "creative-work":0, "group":0, "O":0}
    total = getAmountOfInstances(filepath)
    for entity in percentageOfEntities.keys():
        instances = getInstancesOfEntity(entity, data)
        amount = len(instances)
        AmountOfEntities[entity] = amount
        percentageOfEntities[entity] = (amount/total) * 100

    return AmountOfEntities, percentageOfEntities


if __name__ == '__main__':
    tagsTrain = readTags(r"Data\wnut\wnut17train.conll")
    #print(tagsTrain[0:1000])
    #print(len(tagsTrain))

    tagsTest = readTags(r"Data\wnut\emerging.test.conll")  # error due to encoding
    print(tagsTest[190:199])

    wordTaggedSentencesTrain, entitiesTrain = tokenize(tagsTrain)
    wordTaggedSentencesTest, entitiesTest = tokenize(tagsTest)
    #print(wordTaggedSentencesTrain[0:10])
    #print(entitiesTrain)
    print(wordTaggedSentencesTest[7])
    print(entitiesTest[7])

    posTaggedSentencesTrain = posTag(wordTaggedSentencesTrain)
    posTaggedSentencesTest = posTag(wordTaggedSentencesTest)
    #print(posTaggedSentencesTrain[0:10])
    print(posTaggedSentencesTest[7])

    completeTaggedSentencesTrain = addEntitiyTaggs(posTaggedSentencesTrain, entitiesTrain)
    completeTaggedSentencesTest = addEntitiyTaggs(posTaggedSentencesTest, entitiesTest)
    #print(completeTaggedSentencesTrain[0:10])
    print(completeTaggedSentencesTest[7])

    writeToFile(completeTaggedSentencesTrain, r"Data\Corpus\train\train.conll")
    writeToFile(completeTaggedSentencesTest, r"Data\Corpus\test\test.conll")

    trainInstances, trainPercentages = calculatePercentageOfEntities(r"Data\wnut\wnut17train.conll")
    #print(trainInstances)
    #print(trainPercentages)

    testInstances, testPercentages = calculatePercentageOfEntities(r"Data\wnut\emerging.test.conll")
    #print(testInstances)
    #print(testPercentages)




