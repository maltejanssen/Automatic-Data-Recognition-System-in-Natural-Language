from collections import namedtuple
import nltk
import os


Tag = namedtuple("Tag", ["word", "tag"])

def readTags(file):
    """ Creates a list of tagged words from the corpus
    format: eg. [Tag(word='Empire', tag='B-location'), Tag(word='State', tag='I-location'), Tag(word='Building', tag='I-location')]

    :param file: destination of file from which to extract tags;
    Each line has to have two columns: 1.= word 2.: entity;
    Sentences are to be separated by empty line
    :return: tags: list of read Tags as named tuples, empty tuple signals ending of sentence
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
                tags.append(Tag("", ""))  # append empty tuple to mark sentence ending
    return tags


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

def writeToFile(tagged, filename):
    """ writes pos tagged triplets to file
    -> used so corpus can be read by slightly modified ConllChunkCorpusReader

    :param tagged: list of lists containing pos tagged iob triplets of each sentence in format: (('word', 'POS-Tag'), 'entity')
    :param filename: name of file to write
    """
    # iobTriplets = [[(word, pos, entity) for ((word, pos), entity) in sentence] for sentence in tagged]
    # print(iobTriplets)

    iobTriplets = []
    for sentences in tagged:
        sentence = []
        for triplet in sentences:
            sentence.append((triplet[0][0], triplet[0][1], triplet[1]))
        iobTriplets.append(sentence)
        sentence = []

    scriptDir = os.path.dirname(__file__)
    path = os.path.join(scriptDir, filename)
    with open(path, 'w', encoding='utf-8') as fp:
        #fp.write('\n'.join('{} {} {}'.format(triplet[0], triplet[1], triplet[2]) for triplet in sentence for sentence in iobTriblets))
        for sentence in iobTriplets:
            fp.write("\n".join("{} {} {}".format(triplet[0],triplet[1],triplet[2]) for triplet in sentence))
            fp.write("\n")
            fp.write("\n")



if __name__ == '__main__':
    tagsTrain = readTags(r"Data\wnut17train.conll")
    print(tagsTrain[0:1000])
    print(len(tagsTrain))

    tagsTest = readTags(r"Data\emerging.test.conll")  # error due to encoding
    print(tagsTest[0:10])

    wordTaggedSentencesTrain, entitiesTrain = tokenize(tagsTrain)
    wordTaggedSentencesTest, entitiesTest = tokenize(tagsTest)
    print(wordTaggedSentencesTrain[0:10])
    print(entitiesTrain)

    posTaggedSentencesTrain = posTag(wordTaggedSentencesTrain)
    posTaggedSentencesTest = posTag(wordTaggedSentencesTest)
    print(posTaggedSentencesTrain[0:10])

    completeTaggedSentencesTrain = addEntitiyTaggs(posTaggedSentencesTrain, entitiesTrain)
    completeTaggedSentencesTest = addEntitiyTaggs(posTaggedSentencesTest, entitiesTest)
    print(completeTaggedSentencesTrain[0:10])

    writeToFile(completeTaggedSentencesTrain, r"Data\Corpus\train\train.conll")
    writeToFile(completeTaggedSentencesTest, r"Data\Corpus\test\test.conll")




