import sys, os
import argparse
scriptDir = os.path.dirname(__file__)
path = os.path.join(scriptDir, "reader")
sys.path.insert(0, path)
from reader import ConllChunkCorpusReader






def train(corpusPath, classifier, eval):
    print(corpusPath)
    print(classifier)
    print(eval)

   # buildChunkTree(corpusPath)


def buildChunkTree(corpusPath):
    """ reads a directory and converts its files into chunkTrees
    :param corpusPath: path of corpus(folder)(files to be converted into chunkTrees)
    :return: chunkTrees: chunked Sentences of read Data eg: [Tree('S', [('@paulwalk', 'VB'), ('It', 'PRP'), ("'s", 'VBZ'),...
    """
    reader = ConllChunkCorpusReader(corpusPath, ".*", ['person', 'location', 'corporation', 'product', 'creative-work', 'group'])
    chunkTrees = reader.chunked_sents()
    return chunkTrees





if  __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description='Script that trains NER-Classifiers')
    parser.add_argument("--corpus", default=r"Data\Corpus", help="relative or absolute path to corpus")
    parser.add_argument("--classifier", default="all", help="ClassifierChunker algorithm to use instead of a sequential Tagger based Chunker. Maxent uses the default Maxent training algorithm, either CG or iis.")
    parser.add_argument("--eval", action='store_true', default=False, help="do evaluation")
    args = parser.parse_args()
    train(args.corpus, args.classifier, args.eval)
