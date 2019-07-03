import os
import sys

scriptDir = os.path.dirname(__file__)
path = os.path.join(scriptDir, "reader")
sys.path.insert(0, path)
from .reader.reader import ConllChunkCorpusReader



def buildChunkTree(corpusPath):
    """ reads a directory and converts its files into chunkTrees
    :param corpusPath: path of corpus(folder)(files to be converted into chunkTrees)
    :return: chunkTrees: chunked Sentences of read Data eg: [Tree('S', [('@paulwalk', 'VB'), ('It', 'PRP'), ("'s", 'VBZ'),...
    """
    reader = ConllChunkCorpusReader(corpusPath, ".*", ['person', 'location', 'corporation', 'product', 'creative-work', 'group', 'O'])
    chunkTrees = reader.chunked_sents()
    return chunkTrees


def readSentences(corpusPath):
    reader = ConllChunkCorpusReader(corpusPath, ".*", ['person', 'location', 'corporation', 'product', 'creative-work', 'group', 'O'])
    sentences = reader.sents()
    return sentences


def readWords(corpusPath):
    reader = ConllChunkCorpusReader(corpusPath, ".*",
                                    ['person', 'location', 'corporation', 'product', 'creative-work', 'group', 'O'])
    taggedWords = reader.tagged_words()
    return taggedWords
