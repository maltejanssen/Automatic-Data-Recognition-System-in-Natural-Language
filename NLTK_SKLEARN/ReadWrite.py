from collections import namedtuple
import sys, os
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
