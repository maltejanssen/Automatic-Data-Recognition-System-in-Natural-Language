import numpy as np
from time import time
from nltk import word_tokenize


text = "Peter is an asshole. Fuck me sideways."
print(word_tokenize((text)))



x1 = [[1,2], [1,2,3], [1,2,3,4] ,[1]]
x2 = [[3], [2], [1], [4]]

x1.sort(key=len, reverse=True)
x2.sort(key=len, reverse=True)

print(x1)
print(x2)



x = np.random.randint(-1, 12, (1, 10000000))
y = np.random.randint(-1, 12, (1, 10000000))
x = x[0]
y = y[0]


t1 = time()
mask = (x >= 0)
x = x[mask]
y = y[mask]
print(time()- t1)


x = np.random.randint(-1, 12, (1, 10000000))
y = np.random.randint(-1, 12, (1, 10000000))
x = x[0]
y = y[0]
t1 = time()
idcs = []

for idx, label in enumerate(x):
    if label == -1:
        idcs.append(idx)
goldLabels = np.delete(x, idcs)
predictions = np.delete(y, idcs)
print(time()- t1)


x = [1,2,3,-1,5,6]
x = np.array(x)
mask = (x >= 0)
print(mask)

mask = mask[:,:len(x)]
print(mask)


