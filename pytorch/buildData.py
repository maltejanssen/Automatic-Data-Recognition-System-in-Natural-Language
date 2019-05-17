import csv
import os

def loadDataset(path, type, encoding, delimiter):
    """Loads the dataset in path into memory
    :param str path: path of datafile
    :param str type:  "csv" csv line structure: sentence, word, pos, tag or
    "conll" line structure: word pos tag OR word, tag
    :param str encoding: encoding of file
    :param str delimiter: delimiter between word/tags

    :return dataset: read Data as list of tuples. Each tuple represents a sentence: ([words],[tags])
    """
    with open(path, encoding=encoding) as fp:
        dataset = []
        words, tags = [], []

        # if dataset is in single unpartitioned (not partitioned into train, test, val) csv file
        if type == "csv":
            csv_file = csv.reader(fp, delimiter=delimiter)
            for idx, row in enumerate(csv_file):
                if idx == 0:
                    continue
                sentence, word, pos, tag = row
                if len(sentence) != 0: #!= 0 only if first element of sentence
                    if len(words) > 0:
                        assert len(words) == len(tags)
                        dataset.append((words, tags))
                        words, tags = [], []
                try:
                    word, tag = str(word), str(tag)
                    words.append(word)
                    tags.append(tag)
                except UnicodeDecodeError as e:
                    print("An exception was raised, skipping a word: {}".format(e))
                    pass

        #datset already partitioned
        elif type == "other":
            for line in fp:
                line = line.strip()
                if line:
                    line = line.split(delimiter)
                    # not pos tagged file
                    if len(line) == 2:
                        word = str(line[0])
                        tag = str(line[1])
                        #already pos tagged file
                    elif len(line) == 3:
                        word = str(line[0])
                        tag = str(line[2])
                    else:
                        print(len(line))
                        raise ValueError("unknown format(amount values per line) or wrong delimiter")
                    words.append(word)
                    tags.append(tag)

                # new sentence
                else:
                    if len(words) > 0:
                        dataset.append((words, tags))
                        words, tags = [], []
        else:
            raise ValueError("Type not supported, choose either csv or conll")
    return dataset


def saveDataset(dataset, saveDir):
    """Writes files sentences.txt(containing one sentence per line) and labels.txt(
    containing labels belonging to sentences)in save_dir from dataset

    :param dataset: read Dataset from loadDataset() ([(["a", "cat"], ["O", "O"]), ...])
    :param str saveDir: save location
    """
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)

    with open(os.path.join(saveDir, 'sentences.txt'), 'w', encoding='utf-8') as fSentences:
        with open(os.path.join(saveDir, 'labels.txt'), 'w', encoding='utf-8') as fLables:
            for words, tags in dataset:
                fSentences.write("{}\n".format(" ".join(words)))
                fLables.write("{}\n".format(" ".join(tags)))


# data = load_dataset('data/kaggle/ner_dataset.csv', "csv", "windows-1252", ",")
# print(data[:10])

# train_dataset = data[:int(0.7 * len(data))]
# val_dataset = data[int(0.7 * len(data)): int(0.85 * len(data))]
# test_dataset = data[int(0.85 * len(data)):]

# save_dataset(train_dataset, 'data/kaggle/train')
# save_dataset(val_dataset, 'data/kaggle/val')
# save_dataset(test_dataset, 'data/kaggle/test')

#
trainData = loadDataset("Data/wnut/wnut17train.conll", "other", "utf-8", "\t")
print(trainData[:10])
testData = loadDataset("Data/wnut/emerging.test.annotated", "other", "utf-8", "\t")
valData = loadDataset("Data/wnut/emerging.dev.conll", "other", "utf-8", "\t")

saveDataset(trainData, "Data/train")
saveDataset(testData, "Data/test")
saveDataset(valData, "Data/validation")






