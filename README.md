# NoisyNER - named entity recognition in social media

Repository of the bachelor thesis NoisyNER - named entity recognition in social media. 


## Requirements
Python 3.6.1
+ content in requirements.txt


## Usage
To preprocess the data simply run createData.py and in case of the pytorch classifier also buildVocab.py <br/>
training : python train.py <br/>
For training the pytrch classifier with pre-trained character embedding put the gloVe embedding file (glove.twitter.27B.200d.txt) found at https://nlp.stanford.edu/projects/glove/ in the following folder:  NoisyNER/project/pytorch/Data/embed/glove.twitter.27B  <br/>  
evaluation: python predict.py --eval <br/>
prediction: python predict.py "input sentence" <br/>


## Final Results on the wnut dataset
| Classifier  | F1 score |
| ------------- | ------------- |
| Unigram  | 5.10  |
| Bigram - backoff  | 5.29  |
|Trigram - backoff | 5.29 |
|Decision Tree | 8.15 |
|Bernoulli NB | 17.61 |
|Multinomial NB | 13.91
|SVM | 12.88 |
|GradientBoosting | 10.79 |
|LogisticRegression | 10.79 |
|BI-LSTM CRF | 13.18 |
