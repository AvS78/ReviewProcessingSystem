#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 09:53:31 2020

@author: avs
"""

import numpy as np
import pandas as pd

import pickle


processed=pd.read_pickle("combined_balanced_100k_text_processed_2610.pkl")
print(type(processed),len(processed), processed.head, processed.columns)
print(processed.groupby('dept').size())


#processed by category
# processed_beauty = processed[(processed.dept=="beauty")]
# print(type(processed_beauty),len(processed_beauty), processed_beauty.head, processed_beauty.columns)

# processed_cell = processed[(processed.dept=="cell")]
# print(type(processed_cell),len(processed_cell), processed_cell.head, processed_cell.columns)

# processed_clothing = processed[(processed.dept=="clothing")]
# print(type(processed_clothing),len(processed_clothing))

# processed_electronics = processed[(processed.dept=="electronics")]
# print(type(processed_electronics),len(processed_electronics))

# processed_grocery = processed[(processed.dept=="grocery")]
# print(type(processed_grocery),len(processed_grocery))

# processed_health = processed[(processed.dept=="health")]

# processed_home = processed[(processed.dept=="home")]

# processed_movies = processed[(processed.dept=="movies")]

# processed_pet = processed[(processed.dept=="pet")]

# processed_toys = processed[(processed.dept=="toys")]


#########################################################Preprocessing
import nltk
from nltk.corpus import stopwords
mystopwords=stopwords.words("English") + ['one', 'become', 'get', 'make', 'take']
WNlemma = nltk.WordNetLemmatizer()

def pre_process(text):
    tokens = nltk.word_tokenize(text)
    tokens=[ WNlemma.lemmatize(t.lower()) for t in tokens]
    tokens=[ t for t in tokens if t not in mystopwords]
    tokens = [ t for t in tokens if len(t) >= 3 ]
    return(tokens)


print(type(processed),len(processed), processed.head, processed.columns)

text = processed['processed_with_lemmatize']
toks = text.apply(pre_process)

len(toks)
print(toks.iloc[20])

# Use dictionary (built from corpus) to prepare a DTM (using frequency)
import logging
import gensim 
from gensim import corpora

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

#create the dictionary for the corpus

# Filter off any words with document frequency less than 3, or appearing in NO more than 80% documents
dictionary_review = corpora.Dictionary(toks)
print(dictionary_review)
dictionary_review.filter_extremes(no_below=3, no_above=0.8)
print(dictionary_review[0],len(dictionary_review))

#convert the document according to dictionary for the corpus created above
#dtm here is a list of lists, which is exactly a matrix. doc2bow document to bag of words

dtm_review = [dictionary_review.doc2bow(d) for d in toks]
print(len(dtm_review),type(dtm_review),dtm_review[1])

#Run LDA model fit based on Dictionary based BoW for the document set

lda_review = gensim.models.ldamodel.LdaModel(dtm_review, 
                                             num_topics = 10,
                                             id2word = dictionary_review)

#topics identified at LDA. show them now

lda_review.show_topics()

##Note that different runs result in different but simillar results 
##Label the topics based on representing "topic_words"

dict_rvw_labels = {0: 'game,love, great,fun, kid, set, daughter, play, doll, color', 
               1: 'child, puzzle, play, age, education, letter, song', 
               2: 'year, old, time, month, car, son, first, day',
               3: 'toy, love, like, play, son, little, kid, dog, fun',
               4: 'water, food, product, use, clean, pet, dry, smell, litter, bubble',
               5: 'card, set, battery, sound, figure, track, easy, light',
               6: 'hair, gun, cat, bird, war, head, pony, brush, tail',
               7: 'piece, little, plastic, small, fit, ball, put',
               8: 'player, box, book, board, case, picture',
               9: 'treat, race, color, tea, foam, tray, balance, price, shoulder, bounce'}

# Get the topic distribution of documents

#identify topics for each document

doc_topics = lda_review.get_document_topics(dtm_review)

print(doc_topics[10])

#show the topic distributions for the first 5 docs, 
for i in range(0, 5):
    print(doc_topics[i])

#Select the best topic (with highest score) for each document
from operator import itemgetter
top_topic = [ max(t, key=itemgetter(1))[0] for t in doc_topics ]

print(top_topic[:10])
topics_perDoc = [ dict_rvw_labels[t] for t in top_topic ]




####################################### How many docs in each topic?
labels, counts = np.unique(topics_perDoc, return_counts=True)
print (labels)
print (counts)



###############################################Evaluation 
# Now let's see how well these topics match the actual categories
##### Remember, in actual use of LDA, the documents DON'T come with labeled topics.
##### So nomally we can not access the confusion matrix unless we label some data manually 
# import numpy as np
# from sklearn import metrics
# print(metrics.confusion_matrix(topics_perDoc, correct_labels))
# print(np.mean(topics_perDoc == correct_labels) )
# print(metrics.classification_report(topics_perDoc, correct_labels))

###########################Save LDA model ###########################
from gensim.test.utils import datapath
import pickle

# Save model to disk.
temp_file = datapath("LDA_model_beauty")
print(temp_file)

lda_review.save(temp_file)

# save the dictionary

# Pickle corpus dictionary

a_file = open("dictionary_reviews.pkl", "wb")
pickle.dump(dictionary_review, a_file)
a_file.close()

# Pickle dict_review dictionary
b_file = open("dictionary_labels.pkl", "wb")
pickle.dump(dict_rvw_labels, b_file)
b_file.close()


###########################load pre-trained LDA model & dictionary ###########################

import pickle
from operator import itemgetter

model_lda = '/opt/anaconda3/envs/audio/lib/python3.8/site-packages/gensim/test/test_data/LDA_model_beauty'

lda = gensim.models.ldamodel.LdaModel.load(model_lda)

a_file = open("dictionary_reviews.pkl", "rb")
dictionary_out = pickle.load(a_file)
print(type(dictionary_out))
a_file.close()

b_file = open("dictionary_labels.pkl", "rb")
dictionary_lbl_out = pickle.load(b_file)
print(type(dictionary_lbl_out))
b_file.close()


########################Query, the model using new, unseen documents



review = "I think sentiment prediction might be more accurate if we test within the department itself...so the way people feel negative about phone may not be the same way as about toys.....once you predict department then should use reviews within that department to predict...time hota toh might have been worth checking out"

import topic_modelling_output as rana


test = (rana.review_label(review))

print (test)


# def pre_process_review(text):
#     tokens = nltk.word_tokenize(text)
#     tokens=[ WNlemma.lemmatize(t.lower()) for t in tokens]
#     tokens=[ t for t in tokens if t not in mystopwords]
#     tokens = [ t for t in tokens if len(t) >= 3 ]
#     return(tokens)

# review_tokens = pre_process_review(review)
# print(review_tokens)


# print(type(dictionary_out))


# #convert review document into dictionary based representation
# review_dtm = dictionary_out.doc2bow(review_tokens)
# print(review_dtm)

# # predicted the topic for new review with LDA
# topic_predictions = lda[review_dtm] 
# print(topic_predictions)

# #Identify the top most topic predicted for the review and return that topic back to main program
# top_topic = max(topic_predictions, key=itemgetter(1))[0]
# print(top_topic, dictionary_lbl_out[top_topic])

# return (dictionary_lbl_out[top_topic])

