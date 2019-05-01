import csv
import os

def load_dataset(path, type, encoding, delimiter):
    """Loads dataset into memory from csv file
    type csv line structure: sentence, word, pos, tag
    type conll line structure: word pos tag OR word, tag
    """
    # Open the csv file, need to specify the encoding for python3
    with open(path, encoding=encoding) as fp: #wnut utf-8 #other one "windows-1252"
        dataset = []
        words, tags = [], []

        # dataset in single file
        if type == "csv":
            csv_file = csv.reader(fp, delimiter=delimiter)
            for idx, row in enumerate(csv_file):
                if idx == 0: continue
                sentence, word, pos, tag = row
                # If the first column is non empty it means we reached a new sentence
                if len(sentence) != 0:
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
        elif type == "conll":
            for line in fp:
                line = line.strip()
                if line:
                    line = line.split(delimiter)
                    print(line)
                    print(len(line))
                    print(line[0])
                    if len(line) == 2: #not po tagged file
                        word = str(line[0])
                        tag = str(line[1])
                    elif len(line) == 3: #postagged file
                        word = str(line[0])
                        tag = str(line[2])
                    else:
                        raise ValueError("unknown values per line") #different error???
                    words.append(word)
                    tags.append(tag)

                else: #new sentence
                    if len(words) > 0:
                        dataset.append((words, tags))
                        words, tags = [], []
        else:
            raise ValueError("Type not supported, choose either csv or conll")

    return dataset


def save_dataset(dataset, save_dir):
    """Writes sentences.txt and labels.txt files in save_dir from dataset

    Args:
        dataset: ([(["a", "cat"], ["O", "O"]), ...])
        save_dir: (string)
    """
    # Create directory if it doesn't exist
    print("Saving in {}...".format(save_dir))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Export the dataset
    with open(os.path.join(save_dir, 'sentences.txt'), 'w') as file_sentences:
        with open(os.path.join(save_dir, 'labels.txt'), 'w') as file_labels:
            for words, tags in dataset:
                file_sentences.write("{}\n".format(" ".join(words)))
                file_labels.write("{}\n".format(" ".join(tags)))
    print("- done.")


data = load_dataset('data/kaggle/ner_dataset.csv', "csv", "windows-1252", ",")
print(data[:10])

data2 = load_dataset('data/kaggle/test.conll', "conll", "utf-8", " ")
print(data2[:10])



