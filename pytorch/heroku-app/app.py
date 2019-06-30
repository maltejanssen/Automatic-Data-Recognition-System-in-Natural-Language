from __future__ import unicode_literals
import sys
import os
import logging
from serve import get_model_api
from flask import Flask,render_template,url_for,request, jsonify
from spacy import displacy
import nltk
import tweepy

nltk.download('punkt')
app = Flask(__name__)
model_api = get_model_api()



app.logger.addHandler(logging.StreamHandler(sys.stdout))
app.logger.setLevel(logging.ERROR)

def turnIntoSpacyFormat(predictions, words, text):
    print(predictions, flush=True)
    print(words, flush=True)
    print(text, flush=True)
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


@app.route('/')
def index():
	return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def api():
    if request.method == "POST":

        input_data = request.form['rawtext']
        output_data, alignedData = model_api(input_data)
        alignedWords = alignedData["words"]
        alignedPredictions = alignedData["predictions"]

        words = output_data["words"]
        text = output_data["text"]
        predictions= output_data["predictions"]

        #response = jsonify(output_data)
        #print(alignedPredictions, flush=True)

        if request.values.get('type') == 'image':
            ents = turnIntoSpacyFormat(predictions, words, text)
            inp = {"text": text, "ents": ents, "title": None}
            htmlm = displacy.render(inp, style="ent", manual=True,
                                    options={"colors": {"PERSON": "#1f5a07", "CORPORATION": "#9f0120",
                                                        "CREATIVE-WORK": "#ff2770", "GROUP": "#4e4e94", "LOCATION": "#eae11a",
                                                        "PRODUCT": "#941aea"}})

            return render_template('index.html', text=alignedWords, predictions=alignedPredictions, htmlm=htmlm)
        else:
            return render_template('index.html', text=alignedWords, predictions=alignedPredictions)
    else:
        return render_template('index.html')


@app.route("/fetch", methods=["GET", "POST"])
def fetch():
    print("here", flush=True)
    print('OAUTH_TOKEN' in os.environ)
    authToken = os.environ.get('OAUTH_TOKEN')
    print(authToken, flush=True)
    authSecret = os.environ.get('OAUTH_SECRET')
    key = os.environ.get('CONSUMER_KEY')
    secretKey = os.environ.get('CONSUMER_SECRET')

    auth = tweepy.OAuthHandler(key, secretKey)
    auth.set_access_token(authToken, authSecret)
    api = tweepy.API(auth)

    keyword = "football"
    tweet = api.search(keyword, count=20)
    for t in tweet:
        try:
            text = t.text
            print(text,flush=True)
        except:
            continue
        else:
            break
    return render_template("index.html", tweet=text)



if __name__ == '__main__':
	app.run(debug=True)

