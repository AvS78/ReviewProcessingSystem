
import pandas as pd
from collections import Counter
from nltk import FreqDist
import pickle
import sys
import numpy as np
import numpy
import pandas
import os
import tensorflow
import sklearn
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

processed=pd.read_pickle("combined_balanced_100k_text_processed_2610.pkl")
processed=processed.replace(to_replace=['positive', 'negative'], value=[1,0])
processed.groupby(['sentiment']).agg(['count'])


# using all the data
source_x = processed["processed_text"]
source_y = processed['sentiment']
source_positive=processed.loc[processed['sentiment'] == 1]
source_negative=processed.loc[processed['sentiment'] == 0]
print(source_positive.shape, source_negative.shape)
# make dataset balanced
source_positive=source_positive.sample(n=207959)
print(source_positive.shape)

source=pd.concat([source_positive,source_negative], ignore_index=True)


vect = CountVectorizer()
tfidf = TfidfTransformer()
clf = MultinomialNB()
vectorizer = CountVectorizer(max_features=100000, min_df=5, max_df=0.7)
X=source['processed_with_lemmatize']
vX = vect.fit_transform(X)
tfidfX = tfidf.fit_transform(vX)


x_train, x_test, y_train, y_test=train_test_split(tfidfX, source.sentiment,test_size=0.2, random_state=42, shuffle=True,stratify=source['sentiment'])


model = clf.fit(x_train, y_train)
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


from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC


sgc=SGDClassifier()
model = sgc.fit(x_train, y_train)
y_pred=model.predict(x_test)
testscores = metrics.accuracy_score(y_test,y_pred)
confusion = metrics.confusion_matrix(y_test,y_pred)
print("accuracy:%.2f%%" %(testscores*100))
print(metrics.classification_report(y_test,y_pred,digits=2))
print(confusion)



vcf=sklearn.svm.LinearSVC(C=1.0)
model = vcf.fit(x_train, y_train)
y_pred=model.predict(x_test)
testscores = metrics.accuracy_score(y_test,y_pred)
confusion = metrics.confusion_matrix(y_test,y_pred)
print("accuracy:%.2f%%" %(testscores*100))
print(metrics.classification_report(y_test,y_pred,digits=2))
print(confusion)

