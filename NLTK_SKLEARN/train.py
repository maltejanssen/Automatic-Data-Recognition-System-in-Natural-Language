import os
import argparse
import nltk
from sklearn import ensemble, linear_model, naive_bayes, svm, tree
import pickle
from Classifier import ClassifierChunker
from Util import buildChunkTree
from features import prev_next_pos_iob


classifierOptions = ["decisionTree", "NaiveBayes", "maxent", "sklearnExtraTreesClassifier",
                     "sklearnGradientBoostingClassifier", "sklearnRandomForestClassifier", "sklearnLogisticRegression",
                     "sklearnBernoulliNB", "sklearnMultinomialNB", "sklearnLinearSVC", "sklearnNuSVC", "sklearnSVC",
                     "sklearnDecisionTreeClassifier"]


def train(args):  # def train(corpusPath, classifier, eval):
    """ trains a Classifier based on passed Arguments
    :param args: Arguments passed by user
    """
    if args.classifier not in classifierOptions:
        raise ValueError("classifier %s is not supported" % args.classifier)

    if not os.path.isdir(args.corpus + "\\train"):
        raise ValueError("Corpus doesn't contain training directory")

    trainChunkTrees = buildChunkTree(args.corpus + "\\train")
    # chunk trees have to be converted into trainable format| -> difference to Util.addEntitiyTaggs !!
    trainchunks = chunkTrees2trainChunks(trainChunkTrees)

    # dict.get() not usable for default option because of parameters for switch case functions
    if args.classifier in options:
        tagger = options[args.classifier](trainchunks, args)
    else:
        tagger = nltk.tag.ClassifierBasedTagger(train=trainchunks, feature_detector=prev_next_pos_iob,
                                       classifier_builder=makeClassifier(args))

    nerChunker = ClassifierChunker(trainchunks, tagger)
    safeClassifier(nerChunker, args)

    if args.eval:
        if not os.path.isdir(args.corpus + "\\test"):
            print("no test data for evaluation")
        else:
            evalChunkTrees = buildChunkTree(args.corpus + "\\test")
            # trainChunks = chunkTrees2trainChunks(evalChunkTrees)
            eval = nerChunker.evaluate(evalChunkTrees)
            print(eval)


def uniGram(train, args):
    return nltk.tag.UnigramTagger(train=train)


def biGram(train, args):
    if args.backoff == "True":
        backoff = nltk.tag.UnigramTagger(train=train)
    else:
        backoff = None
    return nltk.tag.BigramTagger(train=train, backoff=backoff)


def triGram(train, args):
    if args.backoff == "True":
        backoff = nltk.tag.UnigramTagger(train=train)
        backoff = nltk.tag.BigramTagger(train=train, backoff=backoff)
    else:
        backoff = None

    return nltk.tag.TrigramTagger(train=train, backoff=backoff)


def naiveBayes(train, args):
    return nltk.tag.ClassifierBasedTagger(train=train, feature_detector=prev_next_pos_iob,
                                 classifier_builder=nltk.classify.NaiveBayesClassifier.train)


options = {"1-gram": uniGram,
           "2-gram": biGram,
           "3-gram": triGram,
           "NaiveBayes": naiveBayes,
           }


def chunkTrees2trainChunks(chunkTrees):
    """converts chunkTrees read by CorpusReader into chunks processable by classifier

    :param chunkTrees: chunkTrees read by CorpusReader
    :return: trainChunks
    """
    tagSents = [nltk.chunk.util.tree2conlltags(sent) for sent in chunkTrees]
    return [[((word, pos), entity) for (word, pos, entity) in sent] for sent in tagSents]


def makeClassifier(args):
    """ configurates classifiers with arguments

    :param trainer: String: Name of classifier
    :param args: Classifier Options
    :return: trainFunction of configurated classifier
    """
    trainArgs = {}

    if args.classifier == "maxent":
        classifierTrain = nltk.classify.MaxentClassifier.train
        trainArgs['max_iter'] = args.maxIter
        trainArgs['min_ll'] = args.minll
        trainArgs['min_lldelta'] = args.minlldelta
    elif args.classifier == "decisionTree":
        classifierTrain = nltk.classify.DecisionTreeClassifier.train
        trainArgs['binary'] = False
        trainArgs['entropy_cutoff'] = args.entropyCutoff
        trainArgs['depth_cutoff'] = args.depthCutoff
        trainArgs['support_cutoff'] = args.supportCutoff

    elif args.classifier == "sklearnExtraTreesClassifier":
        classifierTrain = nltk.classify.scikitlearn.SklearnClassifier(
            ensemble.ExtraTreesClassifier(criterion=args.criterion, max_features=args.maxFeats,
                                          max_depth=args.depthCutoff, n_estimators=args.nEstimators)).train
    elif args.classifier == "sklearnGradientBoostingClassifier":
        classifierTrain = nltk.classify.scikitlearn.SklearnClassifier(
            ensemble.GradientBoostingClassifier(learning_rate=args.learningRate, max_features=args.maxFeats,
                                                max_depth=args.depthCutoff, n_estimators=args.nEstimators)).train
    elif args.classifier == "sklearnRandomForestClassifier":
        classifierTrain = nltk.classify.scikitlearn.SklearnClassifier(
            ensemble.RandomForestClassifier(criterion=args.criterion, max_features=args.maxFeats,
                                            max_depth=args.depthCutoff, n_estimators=args.nEstimators)).train
    elif args.classifier == "sklearnLogisticRegression":
        classifierTrain = nltk.classify.scikitlearn.SklearnClassifier(
            linear_model.LogisticRegression(penalty=args.penalty, C=args.C)).train
    elif args.classifier == "sklearnBernoulliNB":
        classifierTrain = nltk.classify.scikitlearn.SklearnClassifier(naive_bayes.BernoulliNB(alpha=args.alpha)).train
    elif args.classifier == "sklearnMultinomialNB":
        classifierTrain = nltk.classify.scikitlearn.SklearnClassifier(naive_bayes.MultinomialNB(alpha=args.alpha)).train
    elif args.classifier == "sklearnLinearSVC":
        classifierTrain = nltk.classify.scikitlearn.SklearnClassifier(
            svm.LinearSVC(C=args.C, penalty=args.penalty, loss=args.loss)).train
    elif args.classifier == "sklearnNuSVC":
        classifierTrain = nltk.classify.scikitlearn.SklearnClassifier(svm.NuSVC(nu=args.nu, kernel=args.kernel)).train
    elif args.classifier == "sklearnSVC":
        classifierTrain = nltk.classify.scikitlearn.SklearnClassifier(svm.SVC(C=args.C, kernel=args.kernel)).train
    elif args.classifier == "sklearnDecisionTreeClassifier":
        classifierTrain = nltk.classify.scikitlearn.SklearnClassifier(
            tree.DecisionTreeClassifier(criterion=args.criterion, max_features=args.maxFeats,
                                        max_depth=args.depthCutoff)).train

    def trainf(trainFeats):
        return classifierTrain(trainFeats, **trainArgs)
    return trainf


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


def addArguments():
    """ configurates arguments passable to script

    :return: parser
    """
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description='Script that trains NER-Classifiers')
    parser.add_argument("--corpus", default=r"Data\Corpus",
                        help="relative or absolute path to corpus; corpus folder has to contain train and test folder")
    parser.add_argument("--classifier", default="all",
                        help="ClassifierChunker algorithm to use instead of a sequential Tagger based Chunker. Maxent uses the default Maxent training algorithm, either CG or iis.")
    parser.add_argument("--eval", action='store_true', default=True, help="do evaluation")
    parser.add_argument("--backoff", default="True", help="turn on/off backoff functionality for n-grams")

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

    sklearnGroup = parser.add_argument_group('sklearn Classifiers',
                                             'These options are used by the sklearn algorithms')
    sklearnGroup.add_argument('--C', type=float, default=1.0,
                              help='Penalty parameter C of the error term, default is %(default)s')
    sklearnGroup.add_argument('--kernel', default='rbf',
                              choices=['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
                              help='kernel type for support vector machine classifiers, default is %(default)s')
    sklearnGroup.add_argument('--learningRate', type=float, default=0.1,
                              help='learning rate, default is %(default)s')
    sklearnGroup.add_argument('--loss', choices=['l1', 'l2'],
                              default='l2', help='loss function, default is %(default)s')
    sklearnGroup.add_argument('--nEstimators', type=int, default=10,
                              help='Number of trees for Decision Tree ensembles, default is %(default)s')
    sklearnGroup.add_argument('--nu', type=float, default=0.5,
                              help='upper bound on fraction of training errors & lower bound on fraction of support vectors, should be in interval of (0,1], default is %(default)s')
    sklearnGroup.add_argument('--penalty', choices=['l1', 'l2'],
                              default='l2', help='norm for penalization, default is %(default)s')
    sklearnGroup.add_argument('--maxFeats', default="auto",
                              help='maximum number of features to consider while looking for a split')
    sklearnGroup.add_argument('--tfidf', default=False, action='store_true',
                              help='Use TfidfTransformer')
    sklearnGroup.add_argument('--criterion', choices=['gini', 'entropy'],
                              default='gini', help='Split quality function, default is %(default)s')
    sklearnGroup.add_argument('--alpha', type=float, default=1.0,
                              help='smoothing parameter for naive bayes classifiers, default is %(default)s')

    return parser.parse_args()


if __name__ == '__main__':
    args = addArguments()

    if args.classifier == "all":
        for classifier in classifierOptions:
            args.classifier = classifier
            train(args)
    else:
        train(args)
