#!/usr/bin/env python
# coding: utf-8


from gensim.models.doc2vec import Doc2Vec
from nltk.tokenize import word_tokenize
import pandas as pd
import nltk
import math
from nltk.stem.porter import PorterStemmer
from nltk import word_tokenize, FreqDist
from nltk.corpus import stopwords
import re


processed=pd.read_pickle("combined_balanced_100k_text_processed_2610.pkl")
data=processed["processed_with_lemmatize"]
data=data.tolist()
check_back=processed['reviewText']


model= Doc2Vec.load("d2v500e50V.model")
#to find the vector of a document which is not in training data
test_data = word_tokenize("I am not happy with my phone".lower())
v1 = model.infer_vector(test_data)
#print("V1_infer", v1)

# to find most similar doc using tags
similar_doc = model.docvecs.most_similar([v1])
#print(similar_doc)
check_back=processed['reviewText']
print(check_back[int((similar_doc[0][0]))])
# to find vector of doc in training data using tags or in other words, printing the vector of document at index 1 in training data
#print(model.docvecs['1'])

