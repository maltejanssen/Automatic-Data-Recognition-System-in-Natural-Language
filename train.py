import sys, os
import argparse
scriptDir = os.path.dirname(__file__)
path = os.path.join(scriptDir, "reader")
sys.path.insert(0, path)
from reader import ConllChunkCorpusReader
from nltk.chunk.util import tree2conlltags
from nltk.tag import UnigramTagger, BigramTagger, TrigramTagger, ClassifierBasedTagger
from Classifier import prev_next_pos_iob, ClassifierChunker
from nltk.classify import DecisionTreeClassifier, MaxentClassifier, NaiveBayesClassifier, megam
import pickle
from Util import buildChunkTree

classifierOtions =  ["1-gram", "2-gram", "3-gram", "decisionTree", "NaiveBayes", "Maxent"]


def train(args): #def train(corpusPath, classifier, eval):
    """ trains a Classifier based on passed Arguments

    :param args: Arguments passed by user-> see main below
    """
    if args.classifier not in classifierOtions:
        raise ValueError("classifier %s is not supported" % args.classifier)

    if not os.path.isdir(args.corpus + "\\train"):
        raise ValueError("Corpus doesn't contain training directory")
    trainChunkTrees = buildChunkTree(args.corpus+ "\\train")
    trainchunks = chunkTrees2trainChunks(trainChunkTrees)

    tagger = options[args.classifier](trainchunks)
    nerChunker = ClassifierChunker(trainchunks, tagger)
    safeClassifier(nerChunker, args)

    #nerChunker = Classifier.TagChunker(trainchunks)

    if args.eval:
        if not os.path.isdir(args.corpus + "\\test"):
            print("no test data for evaluatio")
        else:
            evalChunkTrees = buildChunkTree(args.corpus + "\\test")
            #trainChunks = chunkTrees2trainChunks(evalChunkTrees)
            eval = nerChunker.evaluate(evalChunkTrees)
            print(eval)


def uniGram(train):
    return UnigramTagger(train=train)

def biGram(train):
    return BigramTagger(train=train)

def triGram(train):
    return TrigramTagger(train=train)

def naiveBayes(train):
    return ClassifierBasedTagger(train=train, feature_detector=prev_next_pos_iob,
                                              classifier_builder=NaiveBayesClassifier.train)

def decisionTree(train):
    return ClassifierBasedTagger(train=train, feature_detector=prev_next_pos_iob,
                                               classifier_builder=makeClassifier("decisionTree", args))

def maxent(train):
    return ClassifierBasedTagger(train=train, feature_detector=prev_next_pos_iob,
                                         classifier_builder=makeClassifier("maxent", args))  # MaxentClassifier.train)



options = {"1-gram" : uniGram,
           "2-gram" : biGram,
           "3-gram" : triGram,
           "decisionTree" : naiveBayes,
           "NaiveBayes" : decisionTree,
           "Maxent" : maxent,
}



def chunkTrees2trainChunks(chunkTrees):
    """converts chunkTrees read by CorpusReader into chunks processable by classifier

    :param chunkTrees: chunkTrees read by CorpusReader
    :return: trainChunks
    """
    tagSents = [tree2conlltags(sent) for sent in chunkTrees]
    return [[((word, pos), entity) for (word, pos ,entity) in sent] for sent in tagSents]


def makeClassifier(trainer, args):
    """ configurates classifiers with arguments

    :param trainer: String: Name of classifier
    :param args: Classifier Options
    :return: trainFunction of configurated classifier
    """
    trainArgs = {}

    if trainer == "maxent":
        classifierTrain = MaxentClassifier.train
        trainArgs['max_iter'] = args.maxIter
        trainArgs['min_ll'] = args.minll
        trainArgs['min_lldelta'] = args.minlldelta
    elif trainer == "decisionTree":
        classifierTrain = DecisionTreeClassifier.train
        trainArgs['binary'] = False
        trainArgs['entropy_cutoff'] = args.entropyCutoff
        trainArgs['depth_cutoff'] = args.depthCutoff
        trainArgs['support_cutoff'] = args.supportCutoff

    def train(trainFeats):
        return classifierTrain(trainFeats, **trainArgs)
    return train


def safeClassifier(chunker, args):
    """ safes(pickles) classifierChunker

    :param chunker: chunker/cLassifier to be safed
    :param args: Arguments containing name of classifier
    """
    dir = os.path.dirname(__file__)
    path = os.path.join(dir, "Classifiers")
    file = os.path.join(path, args.classifier)

    f = open(file, 'wb')
    pickle.dump(chunker, f)
    f.close()



if  __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description='Script that trains NER-Classifiers')
    parser.add_argument("--corpus", default=r"Data\Corpus", help="relative or absolute path to corpus; corpus folder has to contain train and test folder")
    parser.add_argument("--classifier", default="all", help="ClassifierChunker algorithm to use instead of a sequential Tagger based Chunker. Maxent uses the default Maxent training algorithm, either CG or iis.")
    parser.add_argument("--eval", action='store_true', default=False, help="do evaluation")


    maxentGroup = parser.add_argument_group("Maxent Classifier")
    maxentGroup.add_argument("-maxIter", default=10, type=int,
                              help="Terminate after default: %(default)d iterations.")
    maxentGroup.add_argument("--minll", default=0, type=float,
                              help="Terminate after the negative average log-likelihood drops under default: %(default)f")
    maxentGroup.add_argument("--minlldelta", default=0.1, type=float,
                              help="Terminate if a single iteration improvesnlog likelihood by less than default, default is %(default)f")

    decisiontreeGroup = parser.add_argument_group("Decision Tree Classifier")
    decisiontreeGroup.add_argument("--entropyCutoff", default=0.05, type=float,
                                    help="default: 0.05")
    decisiontreeGroup.add_argument("--depthCutoff", default=100, type=int,
                                    help="default: 100")
    decisiontreeGroup.add_argument("--supportCutoff", default=10, type=int,
                                    help="default: 10")

    args = parser.parse_args()

    if args.classifier == "all":
        for classifier in classifierOtions:
            args.classifier = classifier
            train(args)
    else:
        train(args)


