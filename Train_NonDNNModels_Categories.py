
import pandas as pd
from collections import Counter
from nltk import FreqDist
import pickle
import sys
import numpy as np
import numpy
import pandas
import os
import gensim
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

processed=pd.read_pickle("combined_balanced_100k_text_processed_2610.pkl")

source=processed
source.head()

vect = CountVectorizer()
tfidf = TfidfTransformer()
mln = MultinomialNB()
vectorizer = CountVectorizer(max_features=100000, min_df=5, max_df=0.7)
X=source['processed_with_pos']
vX = vect.fit_transform(X)
tfidfX = tfidf.fit_transform(vX)

x_train, x_test, y_train, y_test=train_test_split(tfidfX, source.dept,test_size=0.2, random_state=42, shuffle=True,stratify=source['dept'])

model = mln.fit(x_train, y_train)
y_pred=model.predict(x_test)
testscores = metrics.accuracy_score(y_test,y_pred)
confusion = metrics.confusion_matrix(y_test,y_pred)
print("accuracy:%.2f%%" %(testscores*100))
print(metrics.classification_report(y_test,y_pred,digits=2))
print(confusion)


from sklearn import tree
dt=tree.DecisionTreeClassifier()

model = dt.fit(x_train, y_train)
y_pred=model.predict(x_test)
testscores = metrics.accuracy_score(y_test,y_pred)
confusion = metrics.confusion_matrix(y_test,y_pred)
print("accuracy:%.2f%%" %(testscores*100))
print(metrics.classification_report(y_test,y_pred,digits=2))
print(confusion)


import sklearn
vcf=sklearn.svm.LinearSVC(C=1.0)
model = vcf.fit(x_train, y_train)
y_pred=model.predict(x_test)
testscores = metrics.accuracy_score(y_test,y_pred)
confusion = metrics.confusion_matrix(y_test,y_pred)
print("accuracy:%.2f%%" %(testscores*100))
print(metrics.classification_report(y_test,y_pred,digits=2))
print(confusion)

