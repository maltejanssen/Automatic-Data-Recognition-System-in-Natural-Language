def wordshape(text):
    import re
    t1 = re.sub('[A-Z]', 'X',text)
    t2 = re.sub('[a-z]', 'x', t1)
    return re.sub('[0-9]', 'd', t2)


def extractFeatures(tokens, index, history):
    word, pos = tokens[index]

    if index == 0:
        prevword, prevpos, previob = ('<START>',) * 3
    else:
        prevword, prevpos = tokens[index - 1]
        previob = history[index - 1]

    if index == len(tokens) - 1:
        nextword, nextpos = ('<END>',) * 2
    else:
        nextword, nextpos = tokens[index + 1]

    feats = {
        'word': word,
        'pos': pos,
        'nextpos': nextpos,
        'prevpos': prevpos,
        'previob': previob,
        'word.isdigit()': word.isdigit(),
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        "shape": wordshape(word)
    }
    return feats