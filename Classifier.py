from nltk.chunk import ChunkParserI, conlltags2tree, tree2conlltags
import Util
from nltk.tag import UnigramTagger, BigramTagger, TrigramTagger
from reader.reader import ConllChunkCorpusReader

class Classifier(ChunkParserI):

    def __init__(self, trainSents, tagger, **kwargs):
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

def buildChunkTree(corpusPath):
    reader = ConllChunkCorpusReader(corpusPath, ".*", ['person', 'location', 'corporation', 'product', 'creative-work', 'group'])
    chunkTrees = reader.chunked_sents()
    return chunkTrees



if __name__ == '__main__':
    tagsTrain = Util.readTags(r"Data\wnut\wnut17train.conll")
    tagsTest = Util.readTags(r"Data\wnut\emerging.test.conll")

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

    Util.writeToFile(completeTaggedSentencesTrain, r"Data\Corpus\train\train.conll")
    Util.writeToFile(completeTaggedSentencesTest, r"Data\Corpus\test\test.conll")


    trainChunks = buildChunkTree(r"Data\Corpus\train")
    testChunks = buildChunkTree(r"Data\Corpus\test")

    print(trainChunks)
    print(testChunks)