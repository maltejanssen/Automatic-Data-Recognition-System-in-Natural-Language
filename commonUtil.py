import json

def saveDict(dictionary, path):
    """ Safes dictionary to json file

    :param dict dictionary: dictionary of float castable values
    :param path: Safe path of json file
    """
    with open(path, 'w') as f:
        # json needs float values
        dictionary = {k: float(v) for k, v in dictionary.items()}
        json.dump(dictionary, f, indent=4)