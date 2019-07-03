import os
import argparse
import nltk
from utils import util

nltk.download('averaged_perceptron_tagger')
parser = argparse.ArgumentParser(description='script that prepares data')
parser.add_argument("--corpus", default="Data/wnut/")

if __name__ == '__main__':
    """ adds pos tags to wnut data
    """
    print("building data...")
    #get all tags
    args = parser.parse_args()
    tagsTrain = util.readTags(os.path.join(args.corpus, "train.conll"))
    tagsTest = util.readTags(os.path.join(args.corpus, "test.conll"))
    tagsVal = util.readTags(os.path.join(args.corpus, "val.conll"))

    #tokenise into lists of sentences
    wordTaggedSentencesTrain, entitiesTrain = util.tokenize(tagsTrain)
    wordTaggedSentencesTest, entitiesTest = util.tokenize(tagsTest)
    wordTaggedSentencesVal, entitiesVal = util.tokenize(tagsVal)

    #get postags
    posTaggedSentencesTrain = util.posTag(wordTaggedSentencesTrain)
    posTaggedSentencesTest = util.posTag(wordTaggedSentencesTest)
    posTaggedSentencesVal = util.posTag(wordTaggedSentencesVal)

    #add entities to postagged sentences
    completeTaggedSentencesTrain = util.addEntitiyTaggs(posTaggedSentencesTrain, entitiesTrain)
    completeTaggedSentencesTest = util.addEntitiyTaggs(posTaggedSentencesTest, entitiesTest)
    completeTaggedSentencesVal = util.addEntitiyTaggs(posTaggedSentencesVal, entitiesVal)

    #write triplets to file
    util.writeTripletsToFile(completeTaggedSentencesTrain, r"Data\Corpus\train\train.conll")
    util.writeTripletsToFile(completeTaggedSentencesTest, r"Data\Corpus\test\test.conll")
    util.writeTripletsToFile(completeTaggedSentencesVal, r"Data\Corpus\val\val.conll")
    print("...done.")
