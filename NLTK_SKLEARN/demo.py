import os
import pickle
from nltk import word_tokenize, pos_tag
import ReadWrite
from Util import buildChunkTree, readSentences, tokenize, readTags, readWords
import nltk



# filename = "1-gram"
# path = os.path.join("Classifiers", filename)
# file = open(path, "rb")
# classifier = pickle.load(file)
#
# #print(classifier.eval)
#
# sentence = "Peter Miller went to Madrid last weekend to go shopping at Walmart"
# result = classifier.parse(pos_tag(word_tokenize(sentence)))
# print(result)
# result.pretty_print()
# result.draw()
#
# ReadWrite.writeResultToFile(result)

#TODO readwrite to file filename for eval purposes + model eval not add to filebut new one instead
def eval(model):
    tagsTest = readTags(r"Data\wnut\emerging.test.annotated")
    sentences, entitiesTest = tokenize(tagsTest)
    for idx, sentence in enumerate(sentences):
        result = model.parse(pos_tag(sentence))
        ReadWrite.writeResultToFile(result, gold=entitiesTest[idx])


if __name__ == '__main__':
    filename = "sklearnSVC"
    path = os.path.join("Classifiers", filename)
    file = open(path, "rb")
    classifier = pickle.load(file)
    eval(classifier)



