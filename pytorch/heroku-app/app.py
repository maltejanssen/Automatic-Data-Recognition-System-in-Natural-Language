from __future__ import unicode_literals
import sys
import logging
from serve import getModelApi
from flask import Flask,render_template,url_for,request, jsonify
from spacy import displacy

app = Flask(__name__)
model_api = getModelApi()
app.logger.addHandler(logging.StreamHandler(sys.stdout))
app.logger.setLevel(logging.ERROR)



@app.route('/')
def index():
	return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def api():
    if request.method == "POST":

        input_data = request.form['rawtext']
        output_data, alignedData, ents = model_api(input_data)
        alignedWords = alignedData["words"]
        alignedPredictions = alignedData["predictions"]

        words = output_data["words"]
        text = output_data["text"]
        predictions= output_data["predictions"]

        #response = jsonify(output_data)
        #print(alignedPredictions, flush=True)

        if request.values.get('type') == 'image':
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


if __name__ == '__main__':
	app.run()

