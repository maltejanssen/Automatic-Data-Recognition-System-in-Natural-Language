import os
import pickle
import argparse
import nltk
from nltk import pos_tag
from utils.readWrite import writeResultToFile
from utils.util import tokenize, readTags


parser = argparse.ArgumentParser(description='Script that trains NER-Classifiers')
parser.add_argument("--eval", default=False, action='store_true',
                    help='do evaluation')
parser.add_argument("--modelPath", help="path of the model to do prediction")
parser.add_argument("--sentence", help="sentence to be classified")
nltk.download('punkt')

def eval(model, filename=None):
    tagsTest = readTags(os.path.join("Data/wnut", "test.conll"))
    sentences, entitiesTest = tokenize(tagsTest)
    if filename is not None:
        filename = filename + ".txt"
    for idx, sentence in enumerate(sentences):
        result = model.parse(pos_tag(sentence))
        writeResultToFile(result, gold=entitiesTest[idx], filename=filename)



def predict(model, sentence):
    result = model.parse(pos_tag(nltk.word_tokenize(sentence)))
    print(result)
    result.pretty_print()
    result.draw()


if __name__ == '__main__':
    args = parser.parse_args()

    if args.modelPath == None:
        filename = "NaiveBayes"
        path = os.path.join("Classifiers", filename)
    else:
        path = args.modelPath
    file = open(path, "rb")
    classifier = pickle.load(file)
    if args.eval:
        eval(classifier, filename)
    else:
        assert args.sentence is not None
        predict(classifier, args.sentence)

