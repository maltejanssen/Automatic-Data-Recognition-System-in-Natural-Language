from nltk.chunk import ChunkParserI, conlltags2tree, tree2conlltags
import Util
from nltk.tag import UnigramTagger, BigramTagger, TrigramTagger

class Classifier(ChunkParserI):

    def __init__(self, trainSents, tagger, **kwargs):
        self.tagger = tagger

    def parse(self, sentence):
        chunks = self.tagger.tag(sentence)
        iobTriblets = [(word, pos, entity) for ((word, pos), entity) in chunks]
        return conlltags2tree(iobTriblets)

    def evaluate2(self, testSents):
        return self.evaluate([conlltags2tree([(word, pos, entity) for (word, pos), entity in iobs]) for iobs in testSents])



if __name__ == '__main__':
    tagsTrain = Util.readTags(r"Data\wnut17train.conll")
    tagsTest = Util.readTags(r"Data\emerging.test.conll")

    wordTaggedSentencesTrain, entitiesTrain = Util.tokenize(tagsTrain)
    wordTaggedSentencesTest, entitiesTest = Util.tokenize(tagsTest)

    posTaggedSentencesTrain = Util.posTag(wordTaggedSentencesTrain)
    posTaggedSentencesTest = Util.posTag(wordTaggedSentencesTest)

    completeTaggedSentencesTrain = Util.addEntitiyTaggs(posTaggedSentencesTrain, entitiesTrain)
    completeTaggedSentencesTest = Util.addEntitiyTaggs(posTaggedSentencesTest, entitiesTest)

    unigramTagger = UnigramTagger(train=completeTaggedSentencesTrain)
    bigramTagger = BigramTagger(train=completeTaggedSentencesTrain)
    trigramTagger = TrigramTagger(train=completeTaggedSentencesTrain)

    nerChunkerUnigram = Classifier(completeTaggedSentencesTrain, unigramTagger)
    eval = nerChunkerUnigram.evaluate2(completeTaggedSentencesTest)
    print(eval)