import sys, os
import argparse
scriptDir = os.path.dirname(__file__)
path = os.path.join(scriptDir, "reader")
sys.path.insert(0, path)
from reader import ConllChunkCorpusReader






def train(corpusPath, classifier, eval):
    classifierOtions =  ["1-gram", "2-gram", "3-gram", "decisionTree", "NaiveBayes", "Maxent", "all"]

    if classifier not in classifierOtions:
        raise ValueError("classifier %s is not supported" % classifier)


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

    def addMaxentArgs():
        maxent_group = parser.add_argument_group("Maxent Classifier")
        maxent_group.add_argument("-maxIter", default=10, type=int,
                                  help="Terminate after default: %(default)d iterations.")
        maxent_group.add_argument("--minll", default=0, type=float,
                                  help="Terminate after the negative average log-likelihood drops under default: %(default)f")
        maxent_group.add_argument("--minlldelta", default=0.1, type=float,
                                  help="Terminate if a single iteration improvesnlog likelihood by less than default, default is %(default)f")

    def addDecisionTreeArgs():
        decisiontree_group = parser.add_argument_group("Decision Tree Classifier")
        decisiontree_group.add_argument("--entropyCutoff", default=0.05, type=float,
                                        help="default: 0.05")
        decisiontree_group.add_argument("--depthCutoff", default=100, type=int,
                                        help="default: 100")
        decisiontree_group.add_argument("--supportCutoff", default=10, type=int,
                                        help="default: 10")
    train(args.corpus, args.classifier, args.eval)
