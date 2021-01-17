import pandas as pd
from collections import Counter
import pickle
import sys
import numpy as np
import numpy
import pandas
import os
import re
import nltk
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords

stop = stopwords.words('english')
wnl=nltk.WordNetLemmatizer()

def check_word(text):
    return (re.search(r'^[a-zA-Z][a-z0-9A-Z\._]*$', text) is not None)

def clean_text(review):
    processed_review = []
    review = review.lower()
    review = re.sub(r'((www\.[\S]+)|(https?://[\S]+))', ' URL ', review)
    review = re.sub(r'\.{2,}', ' ', review)
    review = review.strip(' "\'')
    review = re.sub(r'(:\s?\)|:-\)|\(\s?:|\(-:|:\'\))', 'POS_FEELING', review)
    review = re.sub(r'(:\s?D|:-D|x-?D|X-?D)', 'POS_FEELING', review)
    review = re.sub(r'(<3|:\*)', 'POS_FEELING', review)
    review = re.sub(r'(;-?\)|;-?D|\(-?;)', 'POS_FEELING', review)
    review = re.sub(r'(:\s?\(|:-\(|\)\s?:|\)-:)', 'NEG_FEELING', review)
    review = re.sub(r'(:,\(|:\'\(|:"\()', 'NEG_FEELING', review)
    
    review = re.sub(r'\s+', ' ', review)
    review=review.replace(".", " ")
    text = review.split()
    for word in text:

        word = word.strip('\'"?!,.():;')
        word = re.sub(r'(.)\1+', r'\1\1', word)
        word = re.sub(r'(-|\')', '', word)
        if check_word(word):
            if word not in stop:
                
                # process only non stop words

                word=wnl.lemmatize(word)
    processed_review.append(word)

    return ' '.join(processed_review)


def return_category(text):
    test_review=clean_text(text)
    
    processed=pd.read_pickle(r"F:\NUS-NLP Project\dept_dataframe.pkl")
    source=processed
    vect = CountVectorizer()
    tfidf = TfidfTransformer()
    mln = MultinomialNB()
    vectorizer = CountVectorizer(max_features=100000, min_df=5, max_df=0.7)
    X=source['processed_with_lemmatize']
    y=X.tolist()
    y.append(test_review)
    X=pd.Series(y)
    vX = vect.fit_transform(X)
    tfidfX = tfidf.fit_transform(vX)
    check_vector=tfidfX[1000000]
    tfidfX=tfidfX[:1000000]
    x_train, x_test, y_train, y_test=train_test_split(tfidfX, source.dept,test_size=0.2, random_state=42, shuffle=True,stratify=source['dept'])
    model = mln.fit(x_train, y_train)
    result=model.predict(check_vector)
    result=result[0]
    return(result)


# text="i have a kitchen top"
# print(return_category(text))

# processed=pd.read_pickle(r"C:\Users\rbfor\OneDrive\Documents\NLP_Project\combined_balanced_100k_text_processed_2610.pkl")

# list(processed)

# processed_dept=processed[['processed_with_lemmatize','dept']]
# processed_dept.to_pickle(r"C:\Users\rbfor\OneDrive\Documents\NLP_Project\dept_dataframe.pkl")
                         # y_pred=model.predict(x_test)
# testscores = metrics.accuracy_score(y_test,y_pred)
# confusion = metrics.confusion_matrix(y_test,y_pred)
# print("accuracy:%.2f%%" %(testscores*100))
# print(metrics.classification_report(y_test,y_pred,digits=2))
# print(confusion)

# test=[]
# test_review=" I like the kitchen cabinet"
# test.append(test_review)
# vX = vect.fit_transform(test)
# tf_test = tfidf.fit_transform(vX)
# print(tf_test.shape)
# y_test=model.predict(tf_test)

# tfidfX.shape

# y_pred[0]

# tf_test

# len(X)
# X[90]
# type(X)
# y=X.tolist()
# y[90]
# y.append(test_review)
# len(y)
# z=pd.Series(y)
# len(z)
# vXz = vect.fit_transform(z)
# tfidfXz = tfidf.fit_transform(vXz)
# tfidfXz.shape
# tfidfXz[100000].shape
# test=tfidfXz[100000]
# test.shape
# tfidfX=tfidfXz[:100000]
# tfidfX.shape

# test_vector=[]
# test_vector.append(test)
# len(test_vector)
# x_test.shape
# test.shape

# xxx=model.predict(test)
# xxx[0]
# # In[9]:


# from sklearn import tree
# dt=tree.DecisionTreeClassifier()


# # In[10]:


# model = dt.fit(x_train, y_train)
# y_pred=model.predict(x_test)
# testscores = metrics.accuracy_score(y_test,y_pred)
# confusion = metrics.confusion_matrix(y_test,y_pred)
# print("accuracy:%.2f%%" %(testscores*100))
# print(metrics.classification_report(y_test,y_pred,digits=2))
# print(confusion)


# # In[11]:


# import sklearn
# vcf=sklearn.svm.LinearSVC(C=1.0)
# model = vcf.fit(x_train, y_train)
# y_pred=model.predict(x_test)
# testscores = metrics.accuracy_score(y_test,y_pred)
# confusion = metrics.confusion_matrix(y_test,y_pred)
# print("accuracy:%.2f%%" %(testscores*100))
# print(metrics.classification_report(y_test,y_pred,digits=2))
# print(confusion)


# # In[ ]:




