import nltk
import Util
from features import prev_next_pos_iob


class ClassifierChunker(nltk.chunk.ChunkParserI):
    def __init__(self, trainSents, tagger,  **kwargs):
        if type(tagger) is not nltk.tag.sequential.UnigramTagger and type(tagger) is not nltk.tag.sequential.BigramTagger and type(tagger) is not nltk.tag.sequential.TrigramTagger:
            self.featureDetector = tagger.feature_detector
        self.tagger = tagger
        self.eval = None

    def parse(self, sentence):
        chunks = self.tagger.tag(sentence)
        iobTriblets = [(word, pos, entity) for ((word, pos), entity) in chunks]
        return nltk.chunk.conlltags2tree(iobTriblets)

    def evaluate2(self, testSents):
        self.eval = self.evaluate([nltk.chunk.conlltags2tree([(word, pos, entity) for (word, pos), entity in iobs]) for iobs in testSents])
        return self.eval



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
    unigramTagger = nltk.tag.UnigramTagger(train=completeTaggedSentencesTrain)
    bigramTagger = nltk.tag.BigramTagger(train=completeTaggedSentencesTrain)
    trigramTagger = nltk.tag.TrigramTagger(train=completeTaggedSentencesTrain)

    #Gram Taggers
    unigramTagger = nltk.tag.UnigramTagger(train=completeTaggedSentencesTrain)
    bigramTagger = nltk.tag.BigramTagger(train=completeTaggedSentencesTrain)
    trigramTagger = nltk.tag.TrigramTagger(train=completeTaggedSentencesTrain)

    #Unigram
    nerChunkerUnigram = ClassifierChunker(completeTaggedSentencesTrain, unigramTagger)
    evalUnigram = nerChunkerUnigram.evaluate2(completeTaggedSentencesTest)
    print("Unigram:")
    print(evalUnigram)
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


    bigramTaggerBackoff = nltk.tag.BigramTagger(train=completeTaggedSentencesTrain, backoff = unigramTagger)
    trigramTaggerBackoff = nltk.tag.TrigramTagger(train=completeTaggedSentencesTrain, backoff = bigramTaggerBackoff)

    nerChunkerTrigramBackoff = ClassifierChunker(completeTaggedSentencesTrain, trigramTaggerBackoff)
    evalTrigramBackoff= nerChunkerTrigramBackoff.evaluate2(completeTaggedSentencesTest)
    print(evalTrigramBackoff)

    features = prev_next_pos_iob

    #naiveBayes
    # naiveBayersTagger = ClassifierBasedTagger(train=completeTaggedSentencesTrain, feature_detector=features, classifier_builder=NaiveBayesClassifier.train)
    # nerChunkerNaiveBayers = ClassifierChunker(completeTaggedSentencesTrain, naiveBayersTagger)
    # evalNaiveBayers = nerChunkerNaiveBayers.evaluate2(completeTaggedSentencesTest)
    # print(evalNaiveBayers)
    #
    # #decisionTree
    # decisionTreeTagger = ClassifierBasedTagger(train=completeTaggedSentencesTrain, feature_detector=features,classifier_builder=DecisionTreeClassifier.train)
    # nerChunkerDecisionTree = ClassifierChunker(completeTaggedSentencesTrain, decisionTreeTagger)
    # evalDecisionTree = nerChunkerDecisionTree.evaluate2(completeTaggedSentencesTest)
    # print("decision Tree:")
    # print(evalDecisionTree)
    # testChunks = buildChunkTree(r"Data\Corpus\test")
    # #eval2 = nerChunkerNaiveBayers.evaluate(testChunks)
    #
    # comp1 = [conlltags2tree([(word, pos, entity) for (word, pos), entity in iobs]) for iobs in completeTaggedSentencesTest]
    #
    # #(comp1)
    # print(len(comp1))
    # print(len(testChunks))
    # print(testChunks == comp1)
    #
    # #print(completeTaggedSentencesTest)
    # temp3 = [x for x in testChunks if x not in comp1]
    # print(temp3)
    #
    # #print(eval2)











