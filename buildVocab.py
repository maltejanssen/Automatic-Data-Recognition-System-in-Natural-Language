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
    with open(txt_path, "w") as f:
        for token in vocab:
            f.write(token + '\n')


def update_vocab(txt_path, vocab):
    """Update word and tag vocabulary from dataset

    Args:
        txt_path: (string) path to file, one sentence per line
        vocab: (dict or Counter) with update method

    Returns:
        dataset_size: (int) number of elements in the dataset
    """
    with open(txt_path) as f:
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
    with open(json_path, 'w') as f:
        d = {k: v for k, v in d.items()}
        json.dump(d, f, indent=4)



if __name__ == '__main__':
    words = Counter()
    size_train_sentences = update_vocab("Data/train/sentences.txt", words)
    size_dev_sentences = update_vocab("Data/val/sentences.txt", words)
    size_test_sentences = update_vocab("Data/test/sentences.txt", words)

    tags = Counter()
    size_train_tags = update_vocab("Data/train/labels.txt", tags)
    size_dev_tags = update_vocab("Data/val/labels.txt", tags)
    size_test_tags = update_vocab("Data/test/labels.txt", tags)

    assert size_train_sentences == size_train_tags
    assert size_dev_sentences == size_dev_tags
    assert size_test_sentences == size_test_tags

    mostFrequentTags = getFrequent(tags, 10)
    mostFrequentWords = getFrequent(words, 10)

    if PAD_WORD not in words: words.append(PAD_WORD)
    if PAD_TAG not in tags: tags.append(PAD_TAG)
    words.append(UNK_WORD)

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








