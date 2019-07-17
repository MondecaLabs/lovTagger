lovTaggerApi
=============
This small program provides an API made with flask which takes as input a new vocabulary and autotag it with a machine learning model


Installation :
==============
Python 3.7 is needed
The dependencies are in requirements.txt. You can run   ``pip install -r requirements.txt``   to install them locally. It will install flask and other python libraries needed for the machine learning model.

Flask allows to create small web application, with python functions binded to URL with the route() decorator



How does it work :
===================
clf.pkl : Machine learning model (SVM) saved in pickle format, to be loaded by the API 

mlb.pkl : OneHotEncoder model saved in pickle format. It is used to map the output of the ML model which is numeric to the excplicit string labels

api.py : flask api. You can run the api locally by typing  python api.py 
It will host a webpage which can take as input a new vocabulary and return the tags suggested by the machine learning model.
The form function takes POST request from textarea, fileuploader or text form and output the labels predicted by the machine learning model in json format. For now the json result is shown in a new web page
The functions vocab_to_text and process are used to transform the graph inputs into the relevant text used by the ML model. The Tokenizer function is used by the clf object


templates/form.html : main html page composed of the 3 form markups which can take 3 types of inputs. When a POST request occur in one of the 3 forms, the form() function in the api.py script is triggered, thus launching the prediction
