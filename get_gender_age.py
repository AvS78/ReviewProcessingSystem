#!/usr/bin/env python
# coding: utf-8

# In[112]:


import re
import os
import pandas as pd
import pickle, joblib
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import metrics
import string
import numpy as np



# GET GENDER
def return_gender(text):
    blogs_df=pd.read_pickle(r"F:\NUS-NLP Project\blogs_dataframe.pkl")
    check_text=text
    source=blogs_df
    train, test = train_test_split(source, test_size=0.20)
    train_text=train['text'].values
    test_text=test['text'].values
    test_text_list=test_text.tolist()
    test_text_list.append(check_text)
    test_text=np.array(test_text_list)
    
    
    #vt = CountVectorizer(max_features=100)
    vt = TfidfVectorizer(max_features=1000)
    vector = vt.fit_transform(train['text'].values).toarray()
    vector = vt.fit_transform(train_text).toarray()
    test_vector = vt.transform(test_text)
    train_gender=train['gender']
    #test_gender=test['gender']
    #clf = LogisticRegression()
    clf = RandomForestClassifier(n_estimators=100)
    model = clf.fit(vector, train_gender.values)
    prediction = model.predict(test_vector)
    result=prediction[-1]
    return(result)


def return_age_group(text):
    blogs_df=pd.read_pickle(r"F:\NUS-NLP Project\blogs_dataframe.pkl")
    check_text=text
    source=blogs_df
    train, test = train_test_split(source, test_size=0.20)
    train_text=train['text'].values
    test_text=test['text'].values
    test_text_list=test_text.tolist()
    test_text_list.append(check_text)
    test_text=np.array(test_text_list)
    
    
    #vt = CountVectorizer(max_features=100)
    vt = TfidfVectorizer(max_features=1000)
    vector = vt.fit_transform(train['text'].values).toarray()
    vector = vt.fit_transform(train_text).toarray()
    test_vector = vt.transform(test_text)
    train_age_group=train['age_group']
    #test_gender=test['gender']
    #clf = LogisticRegression()
    clf = RandomForestClassifier(n_estimators=100)
    model = clf.fit(vector, train_age_group.values)
    prediction = model.predict(test_vector)
    result=prediction[-1]
    return(result)


# In[18]:


# # multinomial Naive Bayes clasifier
# print ("PREDICTIONS FOR GENDER")

# clf = MultinomialNB()
# model = clf.fit(vector, train_gender.values)
# prediction = model.predict(test_vector)
# test_list=test_gender.to_list()
# testscores = metrics.accuracy_score(prediction,test_list)
# confusion = metrics.confusion_matrix(prediction,test_list)
# print ("Multinomial Naive Bayes Classifier")
# print("accuracy:%.2f%%" %(testscores*100))
# print(metrics.classification_report(prediction,test_list,digits=2))
# print(confusion)

# # Bernoulli clasifier
# clf = BernoulliNB()
# model = clf.fit(vector, train_gender.values)
# prediction = model.predict(test_vector)
# test_list=test_gender.to_list()
# testscores = metrics.accuracy_score(prediction,test_list)
# confusion = metrics.confusion_matrix(prediction,test_list)
# print ("Bernoulli Classifier")
# print("accuracy:%.2f%%" %(testscores*100))
# print(metrics.classification_report(prediction,test_list,digits=2))
# print(confusion)

# # Linear SVC
# clf = LinearSVC()
# model = clf.fit(vector, train_gender.values)
# prediction = model.predict(test_vector)
# test_list=test_gender.to_list()
# testscores = metrics.accuracy_score(prediction,test_list)
# confusion = metrics.confusion_matrix(prediction,test_list)
# print (" Linear Support Vector Classifier")
# print("accuracy:%.2f%%" %(testscores*100))
# print(metrics.classification_report(prediction,test_list,digits=2))
# print(confusion)




# Logistic Regression Clasifier
# clf = LogisticRegression()
# model = clf.fit(vector, train_gender.values)
# prediction = model.predict(test_vector)
# prediction[-1]

# len(prediction)
# test_list=test_gender.to_list()
# testscores = metrics.accuracy_score(prediction,test_list)
# confusion = metrics.confusion_matrix(prediction,test_list)
# print (" Logistic Regression Clasifier")
# print("accuracy:%.2f%%" %(testscores*100))
# print(metrics.classification_report(prediction,test_list,digits=2))
# print(confusion)

# # SGD Classifier
# clf = SGDClassifier()
# model = clf.fit(vector, train_gender.values)
# prediction = model.predict(test_vector)
# test_list=test_gender.to_list()
# testscores = metrics.accuracy_score(prediction,test_list)
# confusion = metrics.confusion_matrix(prediction,test_list)
# print (" SGD Clasifier")
# print("accuracy:%.2f%%" %(testscores*100))
# print(metrics.classification_report(prediction,test_list,digits=2))
# print(confusion)

# # Random Forest Classifier
# clf = RandomForestClassifier(n_estimators=100)
# model = clf.fit(vector, train_gender.values)
# prediction = model.predict(test_vector)
# test_list=test_gender.to_list()
# testscores = metrics.accuracy_score(prediction,test_list)
# confusion = metrics.confusion_matrix(prediction,test_list)
# print (" Random Forest")
# print("accuracy:%.2f%%" %(testscores*100))
# print(metrics.classification_report(prediction,test_list,digits=2))
# print(confusion)


# # In[11]:


# # USE COUNT VECTORIZER FOR AGE GROUP

# train, test = train_test_split(source, test_size=0.20)
# vt = CountVectorizer(max_features=100)
# vector = vt.fit_transform(train['text'].values).toarray()
# test_vector = vt.transform(test['text'].values)
# train_age_group=train['age_group']
# test_age_group=test['age_group']


# # In[12]:


# print ("PREDICTIONS FOR AGE")
# clf = MultinomialNB()
# model = clf.fit(vector, train_age_group.values)
# prediction = model.predict(test_vector)
# test_list=test_age_group.to_list()
# testscores = metrics.accuracy_score(prediction,test_list)
# confusion = metrics.confusion_matrix(prediction,test_list)
# print ("Multinomial Naive Bayes Classifier")
# print("accuracy:%.2f%%" %(testscores*100))
# print(metrics.classification_report(prediction,test_list,digits=2))
# print(confusion)

# # Bernoulli clasifier
# clf = BernoulliNB()
# model = clf.fit(vector, train_age_group.values)
# prediction = model.predict(test_vector)
# test_list=test_age_group.to_list()
# testscores = metrics.accuracy_score(prediction,test_list)
# confusion = metrics.confusion_matrix(prediction,test_list)
# print ("Bernoulli Classifier")
# print("accuracy:%.2f%%" %(testscores*100))
# print(metrics.classification_report(prediction,test_list,digits=2))
# print(confusion)

# # Linear SVC
# clf = LinearSVC()
# model = clf.fit(vector, train_age_group.values)
# prediction = model.predict(test_vector)
# test_list=test_age_group.to_list()
# testscores = metrics.accuracy_score(prediction,test_list)
# confusion = metrics.confusion_matrix(prediction,test_list)
# print (" Linear Support Vector Classifier")
# print("accuracy:%.2f%%" %(testscores*100))
# print(metrics.classification_report(prediction,test_list,digits=2))
# print(confusion)




# # Logistic Regression Clasifier
# clf = LogisticRegression()
# model = clf.fit(vector, train_age_group.values)
# prediction = model.predict(test_vector)
# test_list=test_age_group.to_list()
# testscores = metrics.accuracy_score(prediction,test_list)
# confusion = metrics.confusion_matrix(prediction,test_list)
# print (" Logistic Regression Clasifier")
# print("accuracy:%.2f%%" %(testscores*100))
# print(metrics.classification_report(prediction,test_list,digits=2))
# print(confusion)

# # SGD Classifier
# clf = SGDClassifier()
# model = clf.fit(vector, train_age_group.values)
# prediction = model.predict(test_vector)
# test_list=test_age_group.to_list()
# testscores = metrics.accuracy_score(prediction,test_list)
# confusion = metrics.confusion_matrix(prediction,test_list)
# print (" SGD Clasifier")
# print("accuracy:%.2f%%" %(testscores*100))
# print(metrics.classification_report(prediction,test_list,digits=2))
# print(confusion)

# # Random Forest Classifier
# clf = RandomForestClassifier(n_estimators=100)
# model = clf.fit(vector, train_age_group.values)
# prediction = model.predict(test_vector)
# test_list=test_age_group.to_list()
# testscores = metrics.accuracy_score(prediction,test_list)
# confusion = metrics.confusion_matrix(prediction,test_list)
# print (" Random Forest")
# print("accuracy:%.2f%%" %(testscores*100))
# print(metrics.classification_report(prediction,test_list,digits=2))
# print(confusion)


# # In[14]:


# # USE TF_IDF VECTORIZER FOR GENDER
# train, test = train_test_split(source, test_size=0.20)
# #vt = CountVectorizer(max_features=100)
# vt = TfidfVectorizer(max_features=1000)
# vector = vt.fit_transform(train['text'].values).toarray()
# test_vector = vt.transform(test['text'].values)
# train_gender=train['gender']
# test_gender=test['gender']


# # In[15]:


# # multinomial Naive Bayes clasifier
# print ("PREDICTIONS FOR GENDER")

# clf = MultinomialNB()
# model = clf.fit(vector, train_gender.values)
# prediction = model.predict(test_vector)
# test_list=test_gender.to_list()
# testscores = metrics.accuracy_score(prediction,test_list)
# confusion = metrics.confusion_matrix(prediction,test_list)
# print ("Multinomial Naive Bayes Classifier")
# print("accuracy:%.2f%%" %(testscores*100))
# print(metrics.classification_report(prediction,test_list,digits=2))
# print(confusion)

# # Bernoulli clasifier
# clf = BernoulliNB()
# model = clf.fit(vector, train_gender.values)
# prediction = model.predict(test_vector)
# test_list=test_gender.to_list()
# testscores = metrics.accuracy_score(prediction,test_list)
# confusion = metrics.confusion_matrix(prediction,test_list)
# print ("Bernoulli Classifier")
# print("accuracy:%.2f%%" %(testscores*100))
# print(metrics.classification_report(prediction,test_list,digits=2))
# print(confusion)

# # Linear SVC
# clf = LinearSVC()
# model = clf.fit(vector, train_gender.values)
# prediction = model.predict(test_vector)
# test_list=test_gender.to_list()
# testscores = metrics.accuracy_score(prediction,test_list)
# confusion = metrics.confusion_matrix(prediction,test_list)
# print (" Linear Support Vector Classifier")
# print("accuracy:%.2f%%" %(testscores*100))
# print(metrics.classification_report(prediction,test_list,digits=2))
# print(confusion)




# # Logistic Regression Clasifier
# clf = LogisticRegression()
# model = clf.fit(vector, train_gender.values)
# prediction = model.predict(test_vector)
# test_list=test_gender.to_list()
# testscores = metrics.accuracy_score(prediction,test_list)
# confusion = metrics.confusion_matrix(prediction,test_list)
# print (" Logistic Regression Clasifier")
# print("accuracy:%.2f%%" %(testscores*100))
# print(metrics.classification_report(prediction,test_list,digits=2))
# print(confusion)

# # SGD Classifier
# clf = SGDClassifier()
# model = clf.fit(vector, train_gender.values)
# prediction = model.predict(test_vector)
# test_list=test_gender.to_list()
# testscores = metrics.accuracy_score(prediction,test_list)
# confusion = metrics.confusion_matrix(prediction,test_list)
# print (" SGD Clasifier")
# print("accuracy:%.2f%%" %(testscores*100))
# print(metrics.classification_report(prediction,test_list,digits=2))
# print(confusion)

# # Random Forest Classifier
# clf = RandomForestClassifier(n_estimators=100)
# model = clf.fit(vector, train_gender.values)
# prediction = model.predict(test_vector)
# test_list=test_gender.to_list()
# testscores = metrics.accuracy_score(prediction,test_list)
# confusion = metrics.confusion_matrix(prediction,test_list)
# print (" Random Forest")
# print("accuracy:%.2f%%" %(testscores*100))
# print(metrics.classification_report(prediction,test_list,digits=2))
# print(confusion)


# # In[16]:


# print ("PREDICTIONS FOR AGE")
# clf = MultinomialNB()
# model = clf.fit(vector, train_age_group.values)
# prediction = model.predict(test_vector)
# test_list=test_age_group.to_list()
# testscores = metrics.accuracy_score(prediction,test_list)
# confusion = metrics.confusion_matrix(prediction,test_list)
# print ("Multinomial Naive Bayes Classifier")
# print("accuracy:%.2f%%" %(testscores*100))
# print(metrics.classification_report(prediction,test_list,digits=2))
# print(confusion)

# # Bernoulli clasifier
# clf = BernoulliNB()
# model = clf.fit(vector, train_age_group.values)
# prediction = model.predict(test_vector)
# test_list=test_age_group.to_list()
# testscores = metrics.accuracy_score(prediction,test_list)
# confusion = metrics.confusion_matrix(prediction,test_list)
# print ("Bernoulli Classifier")
# print("accuracy:%.2f%%" %(testscores*100))
# print(metrics.classification_report(prediction,test_list,digits=2))
# print(confusion)

# # Linear SVC
# clf = LinearSVC()
# model = clf.fit(vector, train_age_group.values)
# prediction = model.predict(test_vector)
# test_list=test_age_group.to_list()
# testscores = metrics.accuracy_score(prediction,test_list)
# confusion = metrics.confusion_matrix(prediction,test_list)
# print (" Linear Support Vector Classifier")
# print("accuracy:%.2f%%" %(testscores*100))
# print(metrics.classification_report(prediction,test_list,digits=2))
# print(confusion)




# # Logistic Regression Clasifier
# clf = LogisticRegression()
# model = clf.fit(vector, train_age_group.values)
# prediction = model.predict(test_vector)
# test_list=test_age_group.to_list()
# testscores = metrics.accuracy_score(prediction,test_list)
# confusion = metrics.confusion_matrix(prediction,test_list)
# print (" Logistic Regression Clasifier")
# print("accuracy:%.2f%%" %(testscores*100))
# print(metrics.classification_report(prediction,test_list,digits=2))
# print(confusion)

# # SGD Classifier
# clf = SGDClassifier()
# model = clf.fit(vector, train_age_group.values)
# prediction = model.predict(test_vector)
# test_list=test_age_group.to_list()
# testscores = metrics.accuracy_score(prediction,test_list)
# confusion = metrics.confusion_matrix(prediction,test_list)
# print (" SGD Clasifier")
# print("accuracy:%.2f%%" %(testscores*100))
# print(metrics.classification_report(prediction,test_list,digits=2))
# print(confusion)

# # Random Forest Classifier
# clf = RandomForestClassifier(n_estimators=100)
# model = clf.fit(vector, train_age_group.values)
# prediction = model.predict(test_vector)
# test_list=test_age_group.to_list()
# testscores = metrics.accuracy_score(prediction,test_list)
# confusion = metrics.confusion_matrix(prediction,test_list)
# print (" Random Forest")
# print("accuracy:%.2f%%" %(testscores*100))
# print(metrics.classification_report(prediction,test_list,digits=2))
# print(confusion)


# # In[99]:


# # VIEW CORPUS OF WORDS
# table = str.maketrans('', '', string.punctuation)
# blogs_df=pd.read_pickle("blogs_df.pkl")
# blog_text=blogs_df['text']
# corpus=[]
# for text in blog_text:
#     words=text.split()
#     stripped = [w.translate(table) for w in words]
#     corpus.append(stripped)
# corpus = [item for sublist in corpus for item in sublist]
# data_analysis = nltk.FreqDist(corpus)
# data_analysis.plot(25, cumulative=False)


# # In[114]:


# # Create and generate the word cloud:
# import matplotlib.pyplot as plt

# #convert list to string and generate
# unique_string=(" ").join(corpus)
# wordcloud = WordCloud(width = 1000, height = 500).generate(unique_string)
# plt.figure(figsize=(15,8))
# plt.imshow(wordcloud)
# plt.axis("off")
# #plt.savefig("your_file_name"+".png", bbox_inches='tight')
# plt.show()
# plt.close()


# # In[100]:


# # This shows all stop words in the top. So we have to remove the stop words
# stop = stopwords.words('english')
# corpus_no_stop=[]
# for text in blog_text:
#     words=text.split()
#     stripped = [w.translate(table) for w in words]
#     words_no_stop=[]
#     for word in stripped:
#         word=word.lower()
#         if word not in stop and ' ' not in word:
#             corpus_no_stop.append(word)
# data_analysis = nltk.FreqDist(corpus_no_stop)
# data_analysis.plot(25, cumulative=False)


# # In[115]:


# # Create and generate the word cloud:
# import matplotlib.pyplot as plt

# #convert list to string and generate
# unique_string=(" ").join(corpus_no_stop)
# wordcloud = WordCloud(width = 1000, height = 500).generate(unique_string)
# plt.figure(figsize=(15,8))
# plt.imshow(wordcloud)
# plt.axis("off")
# #plt.savefig("your_file_name"+".png", bbox_inches='tight')
# plt.show()
# plt.close()


# # In[101]:


# # we will remove all stop words from the reviews and run models again
# blogs_cleaned=[]
# for text in blog_text:
#     processed=[]
#     words=text.split()
#     stripped = [w.translate(table) for w in words]
#     for word in stripped:
#         word=word.lower()
#         if word not in stop and ' ' not in word:
#             processed.append(word)
#     blogs_cleaned.append(' '.join(processed))


# # In[103]:


# blogs_df['text']=blogs_cleaned


# # In[107]:


# source_male=blogs_df.loc[blogs_df['gender'] == "male"]
# source_female=blogs_df.loc[blogs_df['gender'] == "female"]
# source_male=source_male.sample(n=31671)
# source=pd.concat([source_male,source_female], ignore_index=True)


# # In[108]:


# # USE COUNT VECTORIZER FOR GENDER
# print (" USING COUNT VECTORIZER")
# train, test = train_test_split(source, test_size=0.20)
# vt = CountVectorizer(max_features=100)
# vector = vt.fit_transform(train['text'].values).toarray()
# test_vector = vt.transform(test['text'].values)
# train_gender=train['gender']
# test_gender=test['gender']
# # multinomial Naive Bayes clasifier
# print ("PREDICTIONS FOR GENDER")

# clf = MultinomialNB()
# model = clf.fit(vector, train_gender.values)
# prediction = model.predict(test_vector)
# test_list=test_gender.to_list()
# testscores = metrics.accuracy_score(prediction,test_list)
# confusion = metrics.confusion_matrix(prediction,test_list)
# print ("Multinomial Naive Bayes Classifier")
# print("accuracy:%.2f%%" %(testscores*100))
# print(metrics.classification_report(prediction,test_list,digits=2))
# print(confusion)

# # Bernoulli clasifier
# clf = BernoulliNB()
# model = clf.fit(vector, train_gender.values)
# prediction = model.predict(test_vector)
# test_list=test_gender.to_list()
# testscores = metrics.accuracy_score(prediction,test_list)
# confusion = metrics.confusion_matrix(prediction,test_list)
# print ("Bernoulli Classifier")
# print("accuracy:%.2f%%" %(testscores*100))
# print(metrics.classification_report(prediction,test_list,digits=2))
# print(confusion)

# # Linear SVC
# clf = LinearSVC()
# model = clf.fit(vector, train_gender.values)
# prediction = model.predict(test_vector)
# test_list=test_gender.to_list()
# testscores = metrics.accuracy_score(prediction,test_list)
# confusion = metrics.confusion_matrix(prediction,test_list)
# print (" Linear Support Vector Classifier")
# print("accuracy:%.2f%%" %(testscores*100))
# print(metrics.classification_report(prediction,test_list,digits=2))
# print(confusion)




# # Logistic Regression Clasifier
# clf = LogisticRegression()
# model = clf.fit(vector, train_gender.values)
# prediction = model.predict(test_vector)
# test_list=test_gender.to_list()
# testscores = metrics.accuracy_score(prediction,test_list)
# confusion = metrics.confusion_matrix(prediction,test_list)
# print (" Logistic Regression Clasifier")
# print("accuracy:%.2f%%" %(testscores*100))
# print(metrics.classification_report(prediction,test_list,digits=2))
# print(confusion)

# # SGD Classifier
# clf = SGDClassifier()
# model = clf.fit(vector, train_gender.values)
# prediction = model.predict(test_vector)
# test_list=test_gender.to_list()
# testscores = metrics.accuracy_score(prediction,test_list)
# confusion = metrics.confusion_matrix(prediction,test_list)
# print (" SGD Clasifier")
# print("accuracy:%.2f%%" %(testscores*100))
# print(metrics.classification_report(prediction,test_list,digits=2))
# print(confusion)

# # Random Forest Classifier
# clf = RandomForestClassifier(n_estimators=100)
# model = clf.fit(vector, train_gender.values)
# prediction = model.predict(test_vector)
# test_list=test_gender.to_list()
# testscores = metrics.accuracy_score(prediction,test_list)
# confusion = metrics.confusion_matrix(prediction,test_list)
# print (" Random Forest")
# print("accuracy:%.2f%%" %(testscores*100))
# print(metrics.classification_report(prediction,test_list,digits=2))
# print(confusion)


# # In[109]:


# # USE COUNT VECTORIZER FOR AGE GROUP

# train, test = train_test_split(source, test_size=0.20)
# vt = CountVectorizer(max_features=100)
# vector = vt.fit_transform(train['text'].values).toarray()
# test_vector = vt.transform(test['text'].values)
# train_age_group=train['age_group']
# test_age_group=test['age_group']

# print ("PREDICTIONS FOR AGE")
# clf = MultinomialNB()
# model = clf.fit(vector, train_age_group.values)
# prediction = model.predict(test_vector)
# test_list=test_age_group.to_list()
# testscores = metrics.accuracy_score(prediction,test_list)
# confusion = metrics.confusion_matrix(prediction,test_list)
# print ("Multinomial Naive Bayes Classifier")
# print("accuracy:%.2f%%" %(testscores*100))
# print(metrics.classification_report(prediction,test_list,digits=2))
# print(confusion)

# # Bernoulli clasifier
# clf = BernoulliNB()
# model = clf.fit(vector, train_age_group.values)
# prediction = model.predict(test_vector)
# test_list=test_age_group.to_list()
# testscores = metrics.accuracy_score(prediction,test_list)
# confusion = metrics.confusion_matrix(prediction,test_list)
# print ("Bernoulli Classifier")
# print("accuracy:%.2f%%" %(testscores*100))
# print(metrics.classification_report(prediction,test_list,digits=2))
# print(confusion)

# # Linear SVC
# clf = LinearSVC()
# model = clf.fit(vector, train_age_group.values)
# prediction = model.predict(test_vector)
# test_list=test_age_group.to_list()
# testscores = metrics.accuracy_score(prediction,test_list)
# confusion = metrics.confusion_matrix(prediction,test_list)
# print (" Linear Support Vector Classifier")
# print("accuracy:%.2f%%" %(testscores*100))
# print(metrics.classification_report(prediction,test_list,digits=2))
# print(confusion)




# # Logistic Regression Clasifier
# clf = LogisticRegression()
# model = clf.fit(vector, train_age_group.values)
# prediction = model.predict(test_vector)
# test_list=test_age_group.to_list()
# testscores = metrics.accuracy_score(prediction,test_list)
# confusion = metrics.confusion_matrix(prediction,test_list)
# print (" Logistic Regression Clasifier")
# print("accuracy:%.2f%%" %(testscores*100))
# print(metrics.classification_report(prediction,test_list,digits=2))
# print(confusion)

# # SGD Classifier
# clf = SGDClassifier()
# model = clf.fit(vector, train_age_group.values)
# prediction = model.predict(test_vector)
# test_list=test_age_group.to_list()
# testscores = metrics.accuracy_score(prediction,test_list)
# confusion = metrics.confusion_matrix(prediction,test_list)
# print (" SGD Clasifier")
# print("accuracy:%.2f%%" %(testscores*100))
# print(metrics.classification_report(prediction,test_list,digits=2))
# print(confusion)

# # Random Forest Classifier
# clf = RandomForestClassifier(n_estimators=100)
# model = clf.fit(vector, train_age_group.values)
# prediction = model.predict(test_vector)
# test_list=test_age_group.to_list()
# testscores = metrics.accuracy_score(prediction,test_list)
# confusion = metrics.confusion_matrix(prediction,test_list)
# print (" Random Forest")
# print("accuracy:%.2f%%" %(testscores*100))
# print(metrics.classification_report(prediction,test_list,digits=2))
# print(confusion)


# # In[110]:


# # USE TF_IDF VECTORIZER FOR GENDER
# train, test = train_test_split(source, test_size=0.20)
# #vt = CountVectorizer(max_features=100)
# vt = TfidfVectorizer(max_features=1000)
# vector = vt.fit_transform(train['text'].values).toarray()
# test_vector = vt.transform(test['text'].values)
# train_gender=train['gender']
# test_gender=test['gender']

# # multinomial Naive Bayes clasifier
# print ("PREDICTIONS FOR GENDER")

# clf = MultinomialNB()
# model = clf.fit(vector, train_gender.values)
# prediction = model.predict(test_vector)
# test_list=test_gender.to_list()
# testscores = metrics.accuracy_score(prediction,test_list)
# confusion = metrics.confusion_matrix(prediction,test_list)
# print ("Multinomial Naive Bayes Classifier")
# print("accuracy:%.2f%%" %(testscores*100))
# print(metrics.classification_report(prediction,test_list,digits=2))
# print(confusion)

# # Bernoulli clasifier
# clf = BernoulliNB()
# model = clf.fit(vector, train_gender.values)
# prediction = model.predict(test_vector)
# test_list=test_gender.to_list()
# testscores = metrics.accuracy_score(prediction,test_list)
# confusion = metrics.confusion_matrix(prediction,test_list)
# print ("Bernoulli Classifier")
# print("accuracy:%.2f%%" %(testscores*100))
# print(metrics.classification_report(prediction,test_list,digits=2))
# print(confusion)

# # Linear SVC
# clf = LinearSVC()
# model = clf.fit(vector, train_gender.values)
# prediction = model.predict(test_vector)
# test_list=test_gender.to_list()
# testscores = metrics.accuracy_score(prediction,test_list)
# confusion = metrics.confusion_matrix(prediction,test_list)
# print (" Linear Support Vector Classifier")
# print("accuracy:%.2f%%" %(testscores*100))
# print(metrics.classification_report(prediction,test_list,digits=2))
# print(confusion)




# # Logistic Regression Clasifier
# clf = LogisticRegression()
# model = clf.fit(vector, train_gender.values)
# prediction = model.predict(test_vector)
# test_list=test_gender.to_list()
# testscores = metrics.accuracy_score(prediction,test_list)
# confusion = metrics.confusion_matrix(prediction,test_list)
# print (" Logistic Regression Clasifier")
# print("accuracy:%.2f%%" %(testscores*100))
# print(metrics.classification_report(prediction,test_list,digits=2))
# print(confusion)

# # SGD Classifier
# clf = SGDClassifier()
# model = clf.fit(vector, train_gender.values)
# prediction = model.predict(test_vector)
# test_list=test_gender.to_list()
# testscores = metrics.accuracy_score(prediction,test_list)
# confusion = metrics.confusion_matrix(prediction,test_list)
# print (" SGD Clasifier")
# print("accuracy:%.2f%%" %(testscores*100))
# print(metrics.classification_report(prediction,test_list,digits=2))
# print(confusion)

# # Random Forest Classifier
# clf = RandomForestClassifier(n_estimators=100)
# model = clf.fit(vector, train_gender.values)
# prediction = model.predict(test_vector)
# test_list=test_gender.to_list()
# testscores = metrics.accuracy_score(prediction,test_list)
# confusion = metrics.confusion_matrix(prediction,test_list)
# print (" Random Forest")
# print("accuracy:%.2f%%" %(testscores*100))
# print(metrics.classification_report(prediction,test_list,digits=2))
# print(confusion)


# # In[111]:


# print ("PREDICTIONS FOR AGE")
# clf = MultinomialNB()
# model = clf.fit(vector, train_age_group.values)
# prediction = model.predict(test_vector)
# test_list=test_age_group.to_list()
# testscores = metrics.accuracy_score(prediction,test_list)
# confusion = metrics.confusion_matrix(prediction,test_list)
# print ("Multinomial Naive Bayes Classifier")
# print("accuracy:%.2f%%" %(testscores*100))
# print(metrics.classification_report(prediction,test_list,digits=2))
# print(confusion)

# # Bernoulli clasifier
# clf = BernoulliNB()
# model = clf.fit(vector, train_age_group.values)
# prediction = model.predict(test_vector)
# test_list=test_age_group.to_list()
# testscores = metrics.accuracy_score(prediction,test_list)
# confusion = metrics.confusion_matrix(prediction,test_list)
# print ("Bernoulli Classifier")
# print("accuracy:%.2f%%" %(testscores*100))
# print(metrics.classification_report(prediction,test_list,digits=2))
# print(confusion)

# # Linear SVC
# clf = LinearSVC()
# model = clf.fit(vector, train_age_group.values)
# prediction = model.predict(test_vector)
# test_list=test_age_group.to_list()
# testscores = metrics.accuracy_score(prediction,test_list)
# confusion = metrics.confusion_matrix(prediction,test_list)
# print (" Linear Support Vector Classifier")
# print("accuracy:%.2f%%" %(testscores*100))
# print(metrics.classification_report(prediction,test_list,digits=2))
# print(confusion)




# # Logistic Regression Clasifier
# clf = LogisticRegression()
# model = clf.fit(vector, train_age_group.values)
# prediction = model.predict(test_vector)
# test_list=test_age_group.to_list()
# testscores = metrics.accuracy_score(prediction,test_list)
# confusion = metrics.confusion_matrix(prediction,test_list)
# print (" Logistic Regression Clasifier")
# print("accuracy:%.2f%%" %(testscores*100))
# print(metrics.classification_report(prediction,test_list,digits=2))
# print(confusion)

# # SGD Classifier
# clf = SGDClassifier()
# model = clf.fit(vector, train_age_group.values)
# prediction = model.predict(test_vector)
# test_list=test_age_group.to_list()
# testscores = metrics.accuracy_score(prediction,test_list)
# confusion = metrics.confusion_matrix(prediction,test_list)
# print (" SGD Clasifier")
# print("accuracy:%.2f%%" %(testscores*100))
# print(metrics.classification_report(prediction,test_list,digits=2))
# print(confusion)

# # Random Forest Classifier
# clf = RandomForestClassifier(n_estimators=100)
# model = clf.fit(vector, train_age_group.values)
# prediction = model.predict(test_vector)
# test_list=test_age_group.to_list()
# testscores = metrics.accuracy_score(prediction,test_list)
# confusion = metrics.confusion_matrix(prediction,test_list)
# print (" Random Forest")
# print("accuracy:%.2f%%" %(testscores*100))
# print(metrics.classification_report(prediction,test_list,digits=2))
# print(confusion)


# # In[ ]:




