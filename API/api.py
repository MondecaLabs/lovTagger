from flask import Flask, request, render_template, flash, redirect, url_for
from werkzeug.utils import secure_filename
from sklearn.externals import joblib
import urllib.request
import rdflib
import json
import re
import validators
import numpy as np
from nltk.corpus import stopwords


#%% Functions for the ML model

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
    print("first char : ", string[0])
    if string[0] == "<":
        form = "xml"
    else:
        form = "n3"
    g = rdflib.Graph()
    g.parse(data = string, format=form)
    text = process(g)
    return text


    
#%% Flask API

UPLOAD_FOLDER = "/uploads/"
ALLOWED_EXTENSIONS = set(["n3", "owl", "ttl"])

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Return true is the filename is accepted for upload
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

"""
@app.route("/", methods=['GET'])
@app.route("/index")
def index():
    return "<h1>Hello, World !</h1>"
"""

@app.route("/", methods=['GET', "POST"])
@app.route("/index", methods=['GET', "POST"])
def form():
    if request.method == "POST":
        print("getting text")
        
        # If input come from text area form        
        if "text" in request.form.keys():
            text = request.form["text"]
        # If inut comes from an uploaded file
        elif "file" in request.files.keys():
            file = request.files["file"]
            if allowed_file(file.filename):
                text = file.read().decode("utf-8")
            else:
                return redirect(request.url)
        # If input comes from uri
        elif "uri" in request.form.keys():
            uri = request.form["uri"]
            if not(validators.url(uri)):
                return redirect(request.url)
            else:
                data = urllib.request.urlopen(uri)
                text = data.read().decode("utf-8")
        # If POST request is sent without good data
        else:
            return redirect(request.url)
        
        print(type(text))
        
        # Process text
        processed_text = vocab_to_text(text)
        
        # Make prediction from text
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
    
    
    
#%%
    