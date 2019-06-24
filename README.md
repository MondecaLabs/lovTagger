# lovTagger

This project aim to provide a classification system which recommend tags for newly submitted ontologies in the LinkedOpenViocabulary Plateform [1]

* Vocabs_process_rdfscomments.ipynb : process the vocabularies in rdf format into text format by extracting all the text information contained in literals following predicates like rdfs:comments or rdfs:label.
* classif_texts_ML.ipynb : Train several machine learning models for the classification of the domain of the vocabularies. The Support Vector Machine (SVM) seems to give the best results
* classif_text_DL.ipynb : Test a Convolutional Neural Network (CNN) for the same task. It does not give scores as good as the machine learning models
* API/ : Implementaion of an API with FLASK which takes as input a new vocabulary and return the recommend tags of the SVM trained on the entire dataset of LOV (version of june 2019)


[1] https://lov.linkeddata.es/dataset/lov
