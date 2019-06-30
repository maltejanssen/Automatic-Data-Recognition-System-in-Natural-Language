import os
import torch
from .utils import util
import model.net2
from .dataLoader import DataLoader
from torch.autograd import Variable
import numpy as np
from nltk import sent_tokenize,word_tokenize
from spacy import displacy
from model.net2 import Net



path = "results\predection"

def predictTestData(params, dataLoader, model):

    if not os.path.exists(path):
        os.mkdir(path)
    path = os.path.join(path, "results.txt")
    if os.path.exists(path):
        os.remove(path)

    dataPath = "Data"
    testData = dataLoader.readData(dataPath, ["test"])
    testData = testData["test"]
    params.test_size = testData["size"]
    testDataGenerator = dataLoader.batchGenerator(testData, params)
    numOfBatches = (params.test_size + 1) // params.batch_size


    for batch in range(numOfBatches):
        sentsBatch, labelsBatch, charsBatch = next(testDataGenerator)
        output = model(sentsBatch, charsBatch)
        crf = True if model.useCrf else False
        if crf:
            mask = torch.autograd.Variable(sentsBatch.data.ne(params.padInd)).float()
            output = model.crflayer.viterbi_decode(output, mask)

        prediction, gold = util.prepareLabels(output, labelsBatch, crf=crf)
        prediction = util.translateIdcToLabel(dataLoader.getidxToTag(), prediction)
        gold = util.translateIdcToLabel(dataLoader.getidxToTag(), gold)

        writeResultstoFile(prediction, gold, sentsBatch, dataLoader.idxToWord, dataLoader.padInd)


def writeResultstoFile(predictions, gold, sentences, idxToWord, padInD):
    sentences = sentences.data.cpu().numpy()
    sentences = sentences.ravel()

    idcs = []
    for idx, label in enumerate(sentences):
        if label == padInD:
            idcs.append(idx)
    sentences = np.delete(sentences, idcs)


    with open(path, "a", encoding='utf-8') as fp:
        for idx, p in enumerate(sentences):
            fp.write("".join("{}\t{}\t{}".format(idxToWord[p], gold[idx], predictions[idx])))
            fp.write("\n")


def predict(text):
    words, wordIdcs, chars = dataLoader.loadSentences(text)
    wordIdcs = Variable(torch.LongTensor(wordIdcs))
    chars = Variable(torch.LongTensor(chars))
    output = model(wordIdcs, chars)


    crf = True if model.useCrf else False
    predictions, _ = prepareLabels(output, crf=crf)
    predictions = translateIdcToLabel(dataLoader.getidxToTag(), predictions)

    return predictions



dataLoader, params = loadData()
model = Net(params, dataLoader.embeddings)
state = torch.load("experiments/1000hiddenDim/best.pth.tar")
model.load_state_dict(state["state_dict"])
model.eval()




pretictTestData(params, dataLoader)




#
#


# def turnIntoSpacyFormat(predictions):
#     entities = []
#     for idx, prediction in enumerate(predictions):
#         if prediction != "O":
#             if prediction[0] == "B":
#                 print(idx)
#                 d = {}
#                 d["start"] = idx
#                 d["label"] = prediction[2:]
#                 d["end"] = idx
#                 i = idx
#                 i += 1
#                 try:
#                     next = predictions[i]
#                 except:
#                     break
#                 while next[0] == "I":
#                     d["end"] = i
#                     try:
#                         i += 1
#                         next = predictions[i]
#                     except:
#                         break
#                 entities.append(d)
#    return entities


def turnIntoSpacyFormat(predictions, words, text):
    print(predictions)
    print(words)
    print(text)
    entities = []
    current = 0
    idx = 0
    while idx < len(predictions):
        if predictions[idx][0] != "O":
            if predictions[idx][0] == "B" or predictions[idx][0] == "I":
                d = {}
                d["start"] = current
                d["label"] = predictions[idx][2:]
                d["end"] = current + len(words[idx])
                current += len(words[idx])
                if current < len(text):
                    if text[current] == " ":
                        current += 1
                idx += 1
            try:
                next = predictions[idx]
            except:
                entities.append(d)
                break
            while next[0] == "I":
                d["end"] = current + len(words[idx])
                current += len(words[idx])
                try:
                    idx += 1
                    next = predictions[idx]
                except:
                    break
                if current < len(text):
                    if text[current] == " ":
                        current += 1
            entities.append(d)
        try:
            current += len(words[idx])
        except:
            break
        if current < len(text):
            if text[current] == " ":
                current += 1
        idx += 1
    return entities


# words, prediction, text =  predict("I am in Spain")
# print(prediction)
# # words = ["I", "went", "to", "this","Pedro"]
# # prediction = ["O","O","O","O","B-pers"]
# ents = turnIntoSpacyFormat(prediction, words, text)
# print(ents)

#

# pred = ["O","O","B-person","I-person","O","B-org","I-org","I-org"]
# ents = turnIntoSpacyFormat(pred)
# print(ents)
#
# words, predictions = predict(sentences)
# print(predictions)
# print(words)
# ents = turnIntoSpacyFormat((predictions))
# print(ents)
#
# sentence = " ".join(words)
# print(sentence)
# inp = {"text": sentence, "ents": ents, "title": None}
# print(inp)
#
obj = displacy.render({
     "text": "hd",
     "ents": [{"start": 0, "end": 1, "label": "product"}], "title": None}, style="ent", manual=True)
# obj = displacy.render({
#     "text": "But Google is starting from behind.",
#     "ents": [{"start": 4, "end": 10, "label": "pers"}],
#     "title": None
#  }, style="ent", manual=True)
# print("html mock")
#print(obj)



# ex = [{"text": "But Google is starting from behind.",
#        "ents": [{"start": 4, "end": 10, "label": "ORG"}],
#        "title": None}]
# html = displacy.render(ex, style="ent", manual=True)


# import tweepy
# import numpy as np
# import pandas as pd
# from random import randint
#
# access_token = '1142817773120294912-2jkEFJILNdYU5u4FrpIXs6cyE88PXd'
# access_token_secret = '3gjD2CUwvXZP1IiDmEmi0z1tXW6tBwTnsLcvi0hT2q9F6'
# consumer_key = '610TSZye3LPkkLkiDZPs07Liq'
# consumer_secret = 'TvFjgomw1PXI43fumSQskKlRUiGzgUNpzAuefVNaYzOPNXuobJ'
#
# auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
# auth.set_access_token(access_token, access_token_secret)
# api = tweepy.API(auth)
#
#
#
#
# keyword = "football"
# tweet = api.search(keyword, count=20)
# rnd = randint(0, 20)
# for t in tweet:
#     try:
#         print(t.text)
#     except:
#         continue
#     else:
#         break
#
# import random
# from urllib import parse
#
# import base64
# import requests
#
# MAX_PAYLOAD_TWEETS = 100
#
# # Customize the user agent used in the Twitter HTTP requests
# API_USER_AGENT = "Noisy-NER"

class OAuthError(Exception):
    """
    Basic exception for OAuth or token errors
    """
    pass


def _get_bearer_token(key, secret):
    """
    OAuth2 function using twitter's "Application Only" auth method.
    With key and secret, make a POST request to get a bearer token that is used in future API calls
    """

    creds = parse.quote_plus(key) + ':' + parse.quote_plus(secret)
    encoded_creds = base64.b64encode(creds.encode('ascii'))

    all_headers = {
        "Authorization": "Basic " + encoded_creds.decode(encoding='UTF-8'),
        "Content-Type": "application/x-www-form-urlencoded;charset=UTF-8",
        "User-Agent": API_USER_AGENT,
    }

    body_content = {
        'grant_type': 'client_credentials'
    }

    resp = requests.post(
        'https://api.twitter.com/oauth2/token',
        data=body_content,
        headers=all_headers
    )

    json = resp.json()

    if json['token_type'] != 'bearer' or 'access_token' not in json:
        raise OAuthError("Did not receive proper bearer token on initial POST")

    return json['access_token']


def _get_tweets(term, bearer_token):
    """
    Given a search term and bearer_token, return a dict of tweets from the API
    """
    all_headers = {
        "Authorization": "Bearer " + bearer_token,
        "User-Agent": API_USER_AGENT
    }

    request_params = {
        'q': term,
        'count': MAX_PAYLOAD_TWEETS,
    }

    resp = requests.get(
        'https://api.twitter.com/1.1/search/tweets.json',
        params=request_params,
        headers=all_headers
    )

    return resp.json()


def get_random_tweet(term, credentials):
    """
    Given a search term and dict credentials, return a random tweet via Twitter search
    """
    bearer_token = _get_bearer_token(
            credentials['consumer_key'],
            credentials['consumer_secret']
    )
    tweets = _get_tweets(term, bearer_token)

    if not tweets.get('statuses'):
        return None

    return random.SystemRandom().choice(tweets['statuses'])


def load_credentials(filepath: str) -> dict:
    """
    Load a file containing key and secret credentials, separated by a line break (\n)
    Returns a dict with the corresponding credentials
    """

    with open(filepath, 'r') as file_resource:
        data = file_resource.read().strip().split('\n')
    return {
        'consumer_key': data[0],
        'consumer_secret': data[1]
    }

# credentials = {}
#
# try:
#     credentials = load_credentials('./credentials')
# except IOError:
#     print('"Credentials" file not found')
#     exit(1)
#
# tweet = get_random_tweet("football", credentials)
#
# if not tweet:
#     print("No tweets were found with that search term. You can try a different term or try again later.")
#     exit(1)


