#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 18:19:16 2020

@author: avs
"""

###########################load pre-trained LDA model & dictionary ###########################

import pickle
from operator import itemgetter

import gensim 
import nltk
from nltk.corpus import stopwords
mystopwords=stopwords.words("English") + ['one', 'become', 'get', 'make', 'take']
WNlemma = nltk.WordNetLemmatizer()


review = "I think sentiment prediction might be more accurate if we test within the department itself...so the way people feel negative about phone may not be the same way as about toys.....once you predict department then should use reviews within that department to predict...time hota toh might have been worth checking out"

def pre_process_review(text):
    tokens = nltk.word_tokenize(text)
    tokens=[ WNlemma.lemmatize(t.lower()) for t in tokens]
    tokens=[ t for t in tokens if t not in mystopwords]
    tokens = [ t for t in tokens if len(t) >= 3 ]
    return(tokens)


def review_label(review):
    
    # model location
    model_lda = '/opt/anaconda3/envs/audio/lib/python3.8/site-packages/gensim/test/test_data/LDA_model_beauty'
    
    # load model
    lda = gensim.models.ldamodel.LdaModel.load(model_lda)
    
    # load dictionaries
    
    #review corpus dictionary
    a_file = open("dictionary_reviews.pkl", "rb")
    dictionary_out = pickle.load(a_file)
    print(type(dictionary_out))
    a_file.close()
    
    #label dictionary
    b_file = open("dictionary_labels.pkl", "rb")
    dictionary_lbl_out = pickle.load(b_file)
    print(type(dictionary_lbl_out))
    b_file.close()

    review_tokens = pre_process_review(review)
    print(review_tokens)
    
    
    print(type(dictionary_out))
    
    
    #convert review document into dictionary based representation
    review_dtm = dictionary_out.doc2bow(review_tokens)
    print(review_dtm)
    
    # predicted the topic for new review with LDA
    topic_predictions = lda[review_dtm] 
    print(topic_predictions)
    
    #Identify the top most topic predicted for the review and return that topic back to main program
    top_topic = max(topic_predictions, key=itemgetter(1))[0]
    print(top_topic, dictionary_lbl_out[top_topic])
    
    return (dictionary_lbl_out[top_topic])