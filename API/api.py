from flask import Flask, request, render_template
from sklearn.externals import joblib
import rdflib
import json
import re
import numpy as np
#nltk.download('stopwords')
from nltk.corpus import stopwords
#from flask import routes

app = Flask(__name__)


# Tokenizer (the pipeline cannot be loaded otherwise)
def Tokenizer(str_input):
    words = re.sub(r"[^A-Za-z0-9\-]", " ", str_input).lower().split()
    words = [word for word in words if word not in stopwords.words('english')]
    return words

# Load model and mlb transformer
clf = joblib.load("clf.pkl")
mlb = joblib.load("mlb.pkl")

# Return the label with the highest value from the probability label vector
def getBestLabel(y_pred_proba):
    # Get one tag for each prediction by the highest value in the predicted vector
    row_maxs = y_pred_proba.max(axis=1, keepdims=True)
    # Indices of maximum value for each row
    ybest = np.where(y_pred_proba == row_maxs, 1, 0)
    # Decode vector
    ybestLabel = mlb.inverse_transform(ybest)
    return ybestLabel


# From a rdflib graph, concatenate all rdfs:comments in a string
def process(rdflib_graph):
    TEXT_SUFFIXES = ["comment", "description", "label", "definition"]
    full_text = ""
    
    for s,p,o in rdflib_graph:
        # remove literals
        #if type(o) != rdflib.term.Literal:
        suf = rdflib_graph.compute_qname(p)[2]
        if suf in TEXT_SUFFIXES:
            text = str(o)
            if len(text) > 0 and text[-1] != ".":
                text += "."
            full_text += text
            full_text += " "
        
        #print(str(o))
        #print(text)
            
    return full_text

# Transform a string of graph file (nt) into the text describing the graph
def vocab_to_text(string):
    if string[0] == "<":
        form = "xml"
    else:
        form = "n3"
    g = rdflib.Graph()
    g.parse(data = string, format=form)
    text  = process(g)
    return text
    
    
#%%

"""
@app.route("/", methods=['GET'])
@app.route("/index")
def index():
    return "<h1>Hello, World !</h1>"
"""

@app.route("/", methods=['GET', "POST"])
@app.route("/index")
@app.route("/form", methods=['GET', "POST"])
def form():
    if request.method == "POST":
        print("getting text")
        text = request.form["text"]
        print(type(text))
        processed_text = vocab_to_text(text)
        print("before pred")
        pred = clf.predict([processed_text])
        pred_prob = clf.predict_proba([processed_text])
        
        # Get results
        pred_labels = mlb.inverse_transform(pred)
        pred_best_label =  getBestLabel(pred_prob)
        
        json_results = json.dumps({"tags":pred_labels, "best_tag":pred_best_label})
        print("after pred")
        return json_results
    else:
        return render_template("form.html")



if __name__ == "__main__":
    app.run(debug=True, port=5002)