from collections import Counter
import json


PAD_WORD = '<pad>'
PAD_TAG = 'O'
UNK_WORD = 'UNK'


def saveVocabToFile(vocab, path):
    """Writes vocabulary into file
    one token/word per line

    :param vocab: iterable that yields token/word
    :param  path: save path
    """
    with open(path, "w", encoding="utf-8") as f:
        for token in vocab:
            f.write(token + '\n')


def updateVocab(path, vocab):
    """ Update vocabulary

    :param str path: path data file
    :param vocab: dict or Counter

    :return size of dataset (number of elements)
    """
    with open(path, encoding='utf-8') as f:
        for i, line in enumerate(f):
            vocab.update(line.strip().split(' '))
    return i+1


def getFrequent(data, minCount):
    """ get words with >= minimum occurrences

    :param data: Counter that represents vocabulary
    :param int minCount: amount of minimum occurrences
    :return: list of words with minimum occurrences
    """
    data = [tok for tok, count in data.items() if count >= minCount]
    return data


def saveVocabInfo(dictionary, path):
    """ Saves dictionary to json file

    :param dictionary: dictionary to be saved
    :param path: path to json file
    """
    with open(path, 'w', encoding="utf-8") as f:
        d = {k: v for k, v in dictionary.items()}
        json.dump(d, f, indent=4)



if __name__ == '__main__':
    words = Counter()
    trainSentencesLength = updateVocab("Data/train/sentences.txt", words)
    evalSentencesLength = updateVocab("Data/val/sentences.txt", words)
    testSentecnesLength = updateVocab("Data/test/sentences.txt", words)

    tags = Counter()
    updateVocab("Data/train/labels.txt", tags)
    updateVocab("Data/val/labels.txt", tags)
    updateVocab("Data/test/labels.txt", tags)

    n = 1
    mostFrequentTags = getFrequent(tags, n)
    mostFrequentWords = getFrequent(words, n)

    if PAD_WORD not in mostFrequentWords:
        mostFrequentWords.append(PAD_WORD)
    if PAD_TAG not in mostFrequentTags:
        mostFrequentTags.append(PAD_TAG)
    mostFrequentWords.append(UNK_WORD)

    saveVocabToFile(mostFrequentWords, "Data/words.txt")
    saveVocabToFile(mostFrequentTags, "Data/tags.txt")

    sizes = {
        'train_size': trainSentencesLength,
        'dev_size': evalSentencesLength,
        'test_size': testSentecnesLength,
        'vocab_size': len(mostFrequentWords),
        'number_of_tags': len(mostFrequentTags),
        'pad_word': PAD_WORD,
        'pad_tag': PAD_TAG,
        'unk_word': UNK_WORD
    }
    saveVocabInfo(sizes, "Data/dataset_params.json")