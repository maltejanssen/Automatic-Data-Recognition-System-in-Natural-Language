charsInSent = []
for w in sentence:  # list comp???
    charsInWord = []
    for c in w:
        if c in self.charMap:
            charsInWord.append(self.charMap[c])
    charsInSent.append(s)
allChars.append(charsInSent)
sentences.append(s)
charsInSent = []
for w in sentence:  # list comp???
    charsInWord = []
    print("wtf")
    print(len(w))
    for c in w:

        print(c)
        if c in self.charMap:
            charsInWord.append(self.charMap[c])
            print(charsInWord)
        else:
            charsInWord.append(self.unkInd)
    charsInSent.append(charsInWord)
    print(charsInSent)
allChars.append(charsInSent)








batchCharsPadded = []
batchCharsLength = []
dAll = []
for sentence in batchChars:
    chars2_sorted = sorted(sentence, key=lambda p: len(p), reverse=True)
    d = {}
    for i, ci in enumerate(sentence):
        for j, cj in enumerate(chars2_sorted):
            if ci == cj and not j in d and not i in d.values():
                d[j] = i
                continue
    while len(chars2_sorted) < longestSentence:
        chars2_sorted.append([])
    dAll.append(d)
    chars2_length = [len(c) for c in chars2_sorted]
    batchCharsLength.append(chars2_length)
    char_maxl = max(chars2_length)
    chars2_mask = np.zeros((len(chars2_sorted), char_maxl), dtype='int')
    for i, c in enumerate(chars2_sorted):
        chars2_mask[i, :chars2_length[i]] = c

    # print(chars2_mask)
    # chars2_mask = Variable(torch.LongTensor(chars2_mask))
    batchCharsPadded.append(chars2_mask)