# lovTagger

This project aims to provide a classification system which recommends tags for newly submitted ontologies in the Linked Open Vocabulary (LOV) Plateform [1]

* Vocabs_process_rdfscomments.ipynb : process the vocabularies from RDF format into text annotations by extracting all the relevant text information contained in literals following predicates like rdfs:comments or rdfs:label
* classif_texts_ML.ipynb : Train several machine learning models for the classification of the subjects of the vocabularies. The Support Vector Machine (SVM) seems to give the best results
* classif_text_DL.ipynb : Test a Convolutional Neural Network (CNN) for the same task. It does not give scores as good as the classical machine learning models
* API/ : Implementation of an API with Flask which takes as input a new vocabulary and return recommended tags with the SVM trained on the entire dataset of LOV (version of june 2019)
* The Dump of the dataset for this test is available a lov.nq

[1] https://lov.linkeddata.es/dataset/lov
