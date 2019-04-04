from nltk.chunk import ChunkParserI, conlltags2tree, tree2conlltags
import Util
from nltk.tag import UnigramTagger, BigramTagger, TrigramTagger, ClassifierBasedTagger

import nltk
from nltk.classify import DecisionTreeClassifier, MaxentClassifier, NaiveBayesClassifier, megam
from nltk_trainer.classification.multi import AvgProbClassifier

class ClassifierChunker(ChunkParserI):
    def __init__(self, trainSents, tagger,  **kwargs):
        if type(tagger) is not nltk.tag.sequential.UnigramTagger and type(tagger) is not nltk.tag.sequential.BigramTagger and type(tagger) is not nltk.tag.sequential.TrigramTagger:
            self.featureDetector = tagger.feature_detector
        self.tagger = tagger

    def parse(self, sentence):
        chunks = self.tagger.tag(sentence)
        iobTriblets = [(word, pos, entity) for ((word, pos), entity) in chunks]
        return conlltags2tree(iobTriblets)

    def evaluate2(self, testSents):
        return self.evaluate([conlltags2tree([(word, pos, entity) for (word, pos), entity in iobs]) for iobs in testSents])


def prev_next_pos_iob(tokens, index, history):
    word, pos = tokens[index]

    if index == 0:
        prevword, prevpos, previob = ('<START>',) * 3
    else:
        prevword, prevpos = tokens[index - 1]
        previob = history[index - 1]

    if index == len(tokens) - 1:
        nextword, nextpos = ('<END>',) * 2
    else:
        nextword, nextpos = tokens[index + 1]

    feats = {
        'word': word,
        'pos': pos,
        'nextword': nextword,
        'nextpos': nextpos,
        'prevword': prevword,
        'prevpos': prevpos,
        'previob': previob
    }
    return feats

def bag_of_words(words):
    return dict([(word, True) for word in words])




if __name__ == '__main__':
    tagsTrain = Util.readTags(r"Data\wnut\wnut17train.conll")
    tagsTest = Util.readTags(r"Data\wnut\emerging.test.conll")

    wordTaggedSentencesTrain, entitiesTrain = Util.tokenize(tagsTrain)
    wordTaggedSentencesTest, entitiesTest = Util.tokenize(tagsTest)

    posTaggedSentencesTrain = Util.posTag(wordTaggedSentencesTrain)
    posTaggedSentencesTest = Util.posTag(wordTaggedSentencesTest)

    completeTaggedSentencesTrain = Util.addEntitiyTaggs(posTaggedSentencesTrain, entitiesTrain)
    completeTaggedSentencesTest = Util.addEntitiyTaggs(posTaggedSentencesTest, entitiesTest)

    # Gram Taggers
    unigramTagger = UnigramTagger(train=completeTaggedSentencesTrain)
    bigramTagger = BigramTagger(train=completeTaggedSentencesTrain)
    trigramTagger = TrigramTagger(train=completeTaggedSentencesTrain)

    #Gram Taggers
    unigramTagger = UnigramTagger(train=completeTaggedSentencesTrain)
    bigramTagger = BigramTagger(train=completeTaggedSentencesTrain)
    trigramTagger = TrigramTagger(train=completeTaggedSentencesTrain)

    #Unigram
    nerChunkerUnigram = ClassifierChunker(completeTaggedSentencesTrain, unigramTagger)
    evalUnigram = nerChunkerUnigram.evaluate2(completeTaggedSentencesTest)
    print("Unigram:")
    print(evalUnigram)

    #Bigram
    nerChunkerBigram = ClassifierChunker(completeTaggedSentencesTrain, bigramTagger)
    evalBigram = nerChunkerBigram.evaluate2(completeTaggedSentencesTest)
    print("Bigram:")
    print(evalBigram)

    #Trigram
    nerChunkerTrigram = ClassifierChunker(completeTaggedSentencesTrain, trigramTagger)
    evalTrigram = nerChunkerTrigram.evaluate2(completeTaggedSentencesTest)
    print("Trigram:")
    print(evalTrigram)

    features = prev_next_pos_iob

    # #naiveBayes
    # naiveBayersTagger = ClassifierBasedTagger(train=completeTaggedSentencesTrain, feature_detector=features, classifier_builder=NaiveBayesClassifier.train)
    # nerChunkerNaiveBayers = ClassifierChunker(completeTaggedSentencesTrain, naiveBayersTagger)
    # evalNaiveBayers = nerChunkerNaiveBayers.evaluate2(completeTaggedSentencesTest)
    # print("naiveBayes:")
    # print(evalNaiveBayers)
    #
    # #decisionTree
    # decisionTreeTagger = ClassifierBasedTagger(train=completeTaggedSentencesTrain, feature_detector=features,classifier_builder=DecisionTreeClassifier.train)
    # nerChunkerDecisionTree = ClassifierChunker(completeTaggedSentencesTrain, decisionTreeTagger)
    # evalDecisionTree = nerChunkerDecisionTree.evaluate2(completeTaggedSentencesTest)
    # print("decision Tree:")
    # print(evalDecisionTree)

    algorithms = ['GIS', 'IIS', 'MEGAM', 'TADM']

    maxEntTagger = ClassifierBasedTagger(train=completeTaggedSentencesTrain, feature_detector=features, classifier_builder=makeClassifier)#MaxentClassifier.train)

    nerChunkerMaxent = ClassifierChunker(completeTaggedSentencesTrain, maxEntTagger)
    evalMaxEnt = nerChunkerMaxent.evaluate2(completeTaggedSentencesTest)
    #print("Maxent" + " " + algo)
    print(evalMaxEnt)


    # maxent
    # for algo in algorithms:
    #     maxEntTagger = ClassifierBasedTagger(train=completeTaggedSentencesTrain, feature_detector=features, classifier_builder=MaxentClassifier.train(algorithm=algo))
    #     nerChunkerMayent = ClassifierChunker(completeTaggedSentencesTrain, maxEntTagger)
    #     evalMaxEnt = nerChunkerMayent.evaluate2(completeTaggedSentencesTest)
    #     print("Maxent" + " " + algo)
    #     print(evalMaxEnt)









