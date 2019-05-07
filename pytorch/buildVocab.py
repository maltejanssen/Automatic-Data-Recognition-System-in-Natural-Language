from collections import Counter
import json


PAD_WORD = '<pad>'
PAD_TAG = 'O'
UNK_WORD = 'UNK'


def saveVocabToFile(vocab, txt_path):
    """Writes one token per line, 0-based line id corresponds to the id of the token.

    Args:
        vocab: (iterable object) yields token
        txt_path: (string) path to vocab file
    """
    with open(txt_path, "w", encoding="utf-8") as f:
        for token in vocab:
            f.write(token + '\n')


def updateVocab(txt_path, vocab):
    """Update word and tag vocabulary from dataset

    Args:
        txt_path: (string) path to file, one sentence per line
        vocab: (dict or Counter) with update method

    Returns:
        dataset_size: (int) number of elements in the dataset
    """
    with open(txt_path, encoding='utf-8') as f:
        for i, line in enumerate(f):
            vocab.update(line.strip().split(' '))

    return i + 1


def getFrequent(data, minCount):
    data = [tok for tok, count in data.items() if count >= minCount]
    return  data


def saveVocabInfo(d, json_path):
    """Saves dict to json file

    Args:
        d: (dict)
        json_path: (string) path to json file
    """
    with open(json_path, 'w', encoding="utf-8") as f:
        d = {k: v for k, v in d.items()}
        json.dump(d, f, indent=4)



if __name__ == '__main__':
    words = Counter()
    size_train_sentences = updateVocab("Data/train/sentences.txt", words)
    size_dev_sentences = updateVocab("Data/validation/sentences.txt", words)
    size_test_sentences = updateVocab("Data/test/sentences.txt", words)

    tags = Counter()
    size_train_tags = updateVocab("Data/train/labels.txt", tags)
    size_dev_tags = updateVocab("Data/validation/labels.txt", tags)
    size_test_tags = updateVocab("Data/test/labels.txt", tags)

    assert size_train_sentences == size_train_tags
    assert size_dev_sentences == size_dev_tags
    assert size_test_sentences == size_test_tags

    mostFrequentTags = getFrequent(tags, 10)
    mostFrequentWords = getFrequent(words, 10)


    if PAD_WORD not in mostFrequentWords: mostFrequentWords.append(PAD_WORD)
    if PAD_TAG not in mostFrequentTags: mostFrequentTags.append(PAD_TAG)
    mostFrequentWords.append(UNK_WORD)

    saveVocabToFile(words, "Data/words.txt")
    saveVocabToFile(tags, "Data/tags.txt")

    sizes = {
        'train_size': size_train_sentences,
        'dev_size': size_dev_sentences,
        'test_size': size_test_sentences,
        'vocab_size': len(words),
        'number_of_tags': len(tags),
        'pad_word': PAD_WORD,
        'pad_tag': PAD_TAG,
        'unk_word': UNK_WORD
    }
    saveVocabInfo(sizes, "Data/dataset_params.json")








