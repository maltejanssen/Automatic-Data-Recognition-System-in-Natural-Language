from __future__ import unicode_literals
import sys
import logging
from serve import get_model_api
from flask import Flask,render_template,url_for,request, jsonify
from spacy import displacy

app = Flask(__name__)
model_api = get_model_api()

# # Web Scraping Pkg
# from bs4 import BeautifulSoup
# # from urllib.request import urlopen
# from urllib import urlopen

# # Fetch Text From Url
# def get_text(url):
# 	page = urlopen(url)
# 	soup = BeautifulSoup(page)
# 	fetched_text = ' '.join(map(lambda p:p.text,soup.find_all('p')))
# 	return fetched_text



app.logger.addHandler(logging.StreamHandler(sys.stdout))
app.logger.setLevel(logging.ERROR)

def turnIntoSpacyFormat(predictions):
    entities = []
    for idx, prediction in enumerate(predictions):
        if prediction != "O":
            if prediction[0] == "B":
                print(idx)
                d = {}
                d["start"] = idx
                d["label"] = prediction[2:]
                d["end"] = idx
                i = idx
                i += 1
                try:
                    next = predictions[i]
                except:
                    break
                while next[0] == "I":
                    d["end"] = i
                    try:
                        i += 1
                        next = predictions[i]
                    except:
                        break
                entities.append(d)
    return entities


@app.route('/')
def index():
    return app.root_path
	#return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def api():
    if request.method == "POST":

        input_data = request.form['rawtext']
        output_data = model_api(input_data)
        alignedText = output_data["words"]
        predictions = output_data["predictions"]
        #response = jsonify(output_data)
        print(predictions, flush=True)

        if request.values.get('type') == 'image':
            text = output_data["text"]
            ents = turnIntoSpacyFormat((predictions))
            inp = {"text": text, "ents": ents, "title": None}
            htmlm = displacy.render(inp, style="ent", manual=True)

            return render_template('index.html', text=alignedText, predictions=predictions, htmlm=htmlm)
        else:
            return render_template('index.html', text=alignedText, predictions=predictions)
    else:
        return render_template('index.html')


if __name__ == '__main__':
	app.run(debug=True)

