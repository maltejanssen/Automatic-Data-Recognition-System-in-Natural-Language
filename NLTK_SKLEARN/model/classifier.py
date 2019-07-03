import nltk
from utils.util import readTags, tokenize, addEntitiyTaggs, posTag


class ClassifierChunker(nltk.chunk.ChunkParserI):
    """ ClassifierChunker class that is used to build all sklearn/scikit-learn classifiers"""
    def __init__(self, trainSents, tagger,  **kwargs):
        if type(tagger) is not nltk.tag.sequential.UnigramTagger and type(tagger) is not nltk.tag.sequential.BigramTagger\
                and type(tagger) is not nltk.tag.sequential.TrigramTagger:
            self.featureDetector = tagger.feature_detector
        self.tagger = tagger
        self.eval = None


    def parse(self, sentence):
        """ parse the given sentence

        :param str sentence: input sentence
        :return: nltk chunktree with classifications
        """
        chunks = self.tagger.tag(sentence)
        iobTriblets = [(word, pos, entity) for ((word, pos), entity) in chunks]
        return nltk.chunk.conlltags2tree(iobTriblets)

    def evaluate2(self, testSents):
        """ evaluaion function

        :param testSents: test sentences
        """
        self.eval = self.evaluate([nltk.chunk.conlltags2tree([(word, pos, entity) for (word, pos), entity in iobs]) for iobs in testSents])
        return self.eval


if __name__ == '__main__':
    #example usages
    tagsTrain = readTags(r"Data\wnut\wnut17train.conll")
    tagsTest = readTags(r"Data\wnut\emerging.test.conll")

    wordTaggedSentencesTrain, entitiesTrain = tokenize(tagsTrain)
    wordTaggedSentencesTest, entitiesTest = tokenize(tagsTest)

    posTaggedSentencesTrain = posTag(wordTaggedSentencesTrain)
    posTaggedSentencesTest = posTag(wordTaggedSentencesTest)

    completeTaggedSentencesTrain = addEntitiyTaggs(posTaggedSentencesTrain, entitiesTrain)
    completeTaggedSentencesTest = addEntitiyTaggs(posTaggedSentencesTest, entitiesTest)

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










