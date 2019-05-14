import os
import pickle
from nltk import word_tokenize, pos_tag
import ReadWrite



filename = "1-gram"
path = os.path.join("Classifiers", filename)
file = open(path, "rb")
classifier = pickle.load(file)

#print(classifier.eval)

sentence = "Peter Miller went to Madrid last weekend to go shopping at Walmart"
result = classifier.parse(pos_tag(word_tokenize(sentence)))
print(result)
result.pretty_print()
result.draw()

ReadWrite.writeResultToFile(result)




