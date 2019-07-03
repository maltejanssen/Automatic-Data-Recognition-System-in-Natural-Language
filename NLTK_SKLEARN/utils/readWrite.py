from collections import namedtuple
import sys, os
import nltk
scriptDir = os.path.dirname(__file__)
path = os.path.join(scriptDir, "reader")
sys.path.insert(0, path)
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


def writeTripletsToFile(tagged, filename):
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


def writeResultToFile(tree, gold=None, filename="results.txt"):
    """ writes predicted results to File

    :param nltk.tree.Tree tree: chunk Tree of sentence
    :param str filename: name of file
    """
    path = os.path.join("results/", "prediction")
    if not os.path.exists(path):
        os.mkdir(path)
    path = os.path.join(path, filename)


    conllTags = nltk.chunk.util.tree2conlltags(tree)
    assert len(conllTags) == len(gold)
    # remove POS Tag
    conllTags = [(word, entity) for (word, pos, entity) in conllTags]
    if gold:
        wordGoldPred = []
        for idx, elem in enumerate(conllTags):
            word, predTag = elem
            goldTag = gold[idx]
            wordGoldPred.append([word, goldTag, predTag])
        #TODO create new file at beginning
        with open(path, "a", encoding='utf-8') as fp:
            fp.write("\n".join("{}\t{}\t{}".format(word, goldTag, predTag) for word, goldTag, predTag in wordGoldPred))
            fp.write("\n")
            fp.write("\n")

    else:
        with open(path, "a", encoding='utf-8') as fp:
            fp.write("\n".join("{}\t{}".format(word, entity) for word, entity in conllTags))
            fp.write("\n")
            fp.write("\n")
