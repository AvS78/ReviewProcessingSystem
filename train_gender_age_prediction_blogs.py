
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
import nltk
from nltk.probability import FreqDist
from nltk.corpus import stopwords
import string
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator


blogs_df=pd.read_pickle("blogs_df.pkl")
blogs_df.head()


blogs_df.groupby(['gender']).agg(['count'])


blogs_df.groupby(['age_group']).agg(['count'])


# GENDER PREDICTION
# balance the dataset
source_male=blogs_df.loc[blogs_df['gender'] == "male"]
source_female=blogs_df.loc[blogs_df['gender'] == "female"]
source_male=source_male.sample(n=31671)


source=pd.concat([source_male,source_female], ignore_index=True)
source.groupby(['gender']).agg(['count'])


# USE COUNT VECTORIZER FOR GENDER
print (" USING COUNT VECTORIZER")
train, test = train_test_split(source, test_size=0.20)
vt = CountVectorizer(max_features=100)
vector = vt.fit_transform(train['text'].values).toarray()
test_vector = vt.transform(test['text'].values)
train_gender=train['gender']
test_gender=test['gender']

# multinomial Naive Bayes clasifier
print ("PREDICTIONS FOR GENDER")

clf = MultinomialNB()
model = clf.fit(vector, train_gender.values)
prediction = model.predict(test_vector)
test_list=test_gender.to_list()
testscores = metrics.accuracy_score(prediction,test_list)
confusion = metrics.confusion_matrix(prediction,test_list)
print ("Multinomial Naive Bayes Classifier")
print("accuracy:%.2f%%" %(testscores*100))
print(metrics.classification_report(prediction,test_list,digits=2))
print(confusion)

# Bernoulli clasifier
clf = BernoulliNB()
model = clf.fit(vector, train_gender.values)
prediction = model.predict(test_vector)
test_list=test_gender.to_list()
testscores = metrics.accuracy_score(prediction,test_list)
confusion = metrics.confusion_matrix(prediction,test_list)
print ("Bernoulli Classifier")
print("accuracy:%.2f%%" %(testscores*100))
print(metrics.classification_report(prediction,test_list,digits=2))
print(confusion)

# Linear SVC
clf = LinearSVC()
model = clf.fit(vector, train_gender.values)
prediction = model.predict(test_vector)
test_list=test_gender.to_list()
testscores = metrics.accuracy_score(prediction,test_list)
confusion = metrics.confusion_matrix(prediction,test_list)
print (" Linear Support Vector Classifier")
print("accuracy:%.2f%%" %(testscores*100))
print(metrics.classification_report(prediction,test_list,digits=2))
print(confusion)




# Logistic Regression Clasifier
clf = LogisticRegression()
model = clf.fit(vector, train_gender.values)
prediction = model.predict(test_vector)
test_list=test_gender.to_list()
testscores = metrics.accuracy_score(prediction,test_list)
confusion = metrics.confusion_matrix(prediction,test_list)
print (" Logistic Regression Clasifier")
print("accuracy:%.2f%%" %(testscores*100))
print(metrics.classification_report(prediction,test_list,digits=2))
print(confusion)

# SGD Classifier
clf = SGDClassifier()
model = clf.fit(vector, train_gender.values)
prediction = model.predict(test_vector)
test_list=test_gender.to_list()
testscores = metrics.accuracy_score(prediction,test_list)
confusion = metrics.confusion_matrix(prediction,test_list)
print (" SGD Clasifier")
print("accuracy:%.2f%%" %(testscores*100))
print(metrics.classification_report(prediction,test_list,digits=2))
print(confusion)

# Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100)
model = clf.fit(vector, train_gender.values)
prediction = model.predict(test_vector)
test_list=test_gender.to_list()
testscores = metrics.accuracy_score(prediction,test_list)
confusion = metrics.confusion_matrix(prediction,test_list)
print (" Random Forest")
print("accuracy:%.2f%%" %(testscores*100))
print(metrics.classification_report(prediction,test_list,digits=2))
print(confusion)


# USE COUNT VECTORIZER FOR AGE GROUP

train, test = train_test_split(source, test_size=0.20)
vt = CountVectorizer(max_features=100)
vector = vt.fit_transform(train['text'].values).toarray()
test_vector = vt.transform(test['text'].values)
train_age_group=train['age_group']
test_age_group=test['age_group']


print ("PREDICTIONS FOR AGE")
clf = MultinomialNB()
model = clf.fit(vector, train_age_group.values)
prediction = model.predict(test_vector)
test_list=test_age_group.to_list()
testscores = metrics.accuracy_score(prediction,test_list)
confusion = metrics.confusion_matrix(prediction,test_list)
print ("Multinomial Naive Bayes Classifier")
print("accuracy:%.2f%%" %(testscores*100))
print(metrics.classification_report(prediction,test_list,digits=2))
print(confusion)

# Bernoulli clasifier
clf = BernoulliNB()
model = clf.fit(vector, train_age_group.values)
prediction = model.predict(test_vector)
test_list=test_age_group.to_list()
testscores = metrics.accuracy_score(prediction,test_list)
confusion = metrics.confusion_matrix(prediction,test_list)
print ("Bernoulli Classifier")
print("accuracy:%.2f%%" %(testscores*100))
print(metrics.classification_report(prediction,test_list,digits=2))
print(confusion)

# Linear SVC
clf = LinearSVC()
model = clf.fit(vector, train_age_group.values)
prediction = model.predict(test_vector)
test_list=test_age_group.to_list()
testscores = metrics.accuracy_score(prediction,test_list)
confusion = metrics.confusion_matrix(prediction,test_list)
print (" Linear Support Vector Classifier")
print("accuracy:%.2f%%" %(testscores*100))
print(metrics.classification_report(prediction,test_list,digits=2))
print(confusion)




# Logistic Regression Clasifier
clf = LogisticRegression()
model = clf.fit(vector, train_age_group.values)
prediction = model.predict(test_vector)
test_list=test_age_group.to_list()
testscores = metrics.accuracy_score(prediction,test_list)
confusion = metrics.confusion_matrix(prediction,test_list)
print (" Logistic Regression Clasifier")
print("accuracy:%.2f%%" %(testscores*100))
print(metrics.classification_report(prediction,test_list,digits=2))
print(confusion)

# SGD Classifier
clf = SGDClassifier()
model = clf.fit(vector, train_age_group.values)
prediction = model.predict(test_vector)
test_list=test_age_group.to_list()
testscores = metrics.accuracy_score(prediction,test_list)
confusion = metrics.confusion_matrix(prediction,test_list)
print (" SGD Clasifier")
print("accuracy:%.2f%%" %(testscores*100))
print(metrics.classification_report(prediction,test_list,digits=2))
print(confusion)

# Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100)
model = clf.fit(vector, train_age_group.values)
prediction = model.predict(test_vector)
test_list=test_age_group.to_list()
testscores = metrics.accuracy_score(prediction,test_list)
confusion = metrics.confusion_matrix(prediction,test_list)
print (" Random Forest")
print("accuracy:%.2f%%" %(testscores*100))
print(metrics.classification_report(prediction,test_list,digits=2))
print(confusion)


# USE TF_IDF VECTORIZER FOR GENDER
train, test = train_test_split(source, test_size=0.20)
#vt = CountVectorizer(max_features=100)
vt = TfidfVectorizer(max_features=1000)
vector = vt.fit_transform(train['text'].values).toarray()
test_vector = vt.transform(test['text'].values)
train_gender=train['gender']
test_gender=test['gender']


# multinomial Naive Bayes clasifier
print ("PREDICTIONS FOR GENDER")

clf = MultinomialNB()
model = clf.fit(vector, train_gender.values)
prediction = model.predict(test_vector)
test_list=test_gender.to_list()
testscores = metrics.accuracy_score(prediction,test_list)
confusion = metrics.confusion_matrix(prediction,test_list)
print ("Multinomial Naive Bayes Classifier")
print("accuracy:%.2f%%" %(testscores*100))
print(metrics.classification_report(prediction,test_list,digits=2))
print(confusion)

# Bernoulli clasifier
clf = BernoulliNB()
model = clf.fit(vector, train_gender.values)
prediction = model.predict(test_vector)
test_list=test_gender.to_list()
testscores = metrics.accuracy_score(prediction,test_list)
confusion = metrics.confusion_matrix(prediction,test_list)
print ("Bernoulli Classifier")
print("accuracy:%.2f%%" %(testscores*100))
print(metrics.classification_report(prediction,test_list,digits=2))
print(confusion)

# Linear SVC
clf = LinearSVC()
model = clf.fit(vector, train_gender.values)
prediction = model.predict(test_vector)
test_list=test_gender.to_list()
testscores = metrics.accuracy_score(prediction,test_list)
confusion = metrics.confusion_matrix(prediction,test_list)
print (" Linear Support Vector Classifier")
print("accuracy:%.2f%%" %(testscores*100))
print(metrics.classification_report(prediction,test_list,digits=2))
print(confusion)




# Logistic Regression Clasifier
clf = LogisticRegression()
model = clf.fit(vector, train_gender.values)
prediction = model.predict(test_vector)
test_list=test_gender.to_list()
testscores = metrics.accuracy_score(prediction,test_list)
confusion = metrics.confusion_matrix(prediction,test_list)
print (" Logistic Regression Clasifier")
print("accuracy:%.2f%%" %(testscores*100))
print(metrics.classification_report(prediction,test_list,digits=2))
print(confusion)

# SGD Classifier
clf = SGDClassifier()
model = clf.fit(vector, train_gender.values)
prediction = model.predict(test_vector)
test_list=test_gender.to_list()
testscores = metrics.accuracy_score(prediction,test_list)
confusion = metrics.confusion_matrix(prediction,test_list)
print (" SGD Clasifier")
print("accuracy:%.2f%%" %(testscores*100))
print(metrics.classification_report(prediction,test_list,digits=2))
print(confusion)

# Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100)
model = clf.fit(vector, train_gender.values)
prediction = model.predict(test_vector)
test_list=test_gender.to_list()
testscores = metrics.accuracy_score(prediction,test_list)
confusion = metrics.confusion_matrix(prediction,test_list)
print (" Random Forest")
print("accuracy:%.2f%%" %(testscores*100))
print(metrics.classification_report(prediction,test_list,digits=2))
print(confusion)


print ("PREDICTIONS FOR AGE")
clf = MultinomialNB()
model = clf.fit(vector, train_age_group.values)
prediction = model.predict(test_vector)
test_list=test_age_group.to_list()
testscores = metrics.accuracy_score(prediction,test_list)
confusion = metrics.confusion_matrix(prediction,test_list)
print ("Multinomial Naive Bayes Classifier")
print("accuracy:%.2f%%" %(testscores*100))
print(metrics.classification_report(prediction,test_list,digits=2))
print(confusion)

# Bernoulli clasifier
clf = BernoulliNB()
model = clf.fit(vector, train_age_group.values)
prediction = model.predict(test_vector)
test_list=test_age_group.to_list()
testscores = metrics.accuracy_score(prediction,test_list)
confusion = metrics.confusion_matrix(prediction,test_list)
print ("Bernoulli Classifier")
print("accuracy:%.2f%%" %(testscores*100))
print(metrics.classification_report(prediction,test_list,digits=2))
print(confusion)

# Linear SVC
clf = LinearSVC()
model = clf.fit(vector, train_age_group.values)
prediction = model.predict(test_vector)
test_list=test_age_group.to_list()
testscores = metrics.accuracy_score(prediction,test_list)
confusion = metrics.confusion_matrix(prediction,test_list)
print (" Linear Support Vector Classifier")
print("accuracy:%.2f%%" %(testscores*100))
print(metrics.classification_report(prediction,test_list,digits=2))
print(confusion)




# Logistic Regression Clasifier
clf = LogisticRegression()
model = clf.fit(vector, train_age_group.values)
prediction = model.predict(test_vector)
test_list=test_age_group.to_list()
testscores = metrics.accuracy_score(prediction,test_list)
confusion = metrics.confusion_matrix(prediction,test_list)
print (" Logistic Regression Clasifier")
print("accuracy:%.2f%%" %(testscores*100))
print(metrics.classification_report(prediction,test_list,digits=2))
print(confusion)

# SGD Classifier
clf = SGDClassifier()
model = clf.fit(vector, train_age_group.values)
prediction = model.predict(test_vector)
test_list=test_age_group.to_list()
testscores = metrics.accuracy_score(prediction,test_list)
confusion = metrics.confusion_matrix(prediction,test_list)
print (" SGD Clasifier")
print("accuracy:%.2f%%" %(testscores*100))
print(metrics.classification_report(prediction,test_list,digits=2))
print(confusion)

# Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100)
model = clf.fit(vector, train_age_group.values)
prediction = model.predict(test_vector)
test_list=test_age_group.to_list()
testscores = metrics.accuracy_score(prediction,test_list)
confusion = metrics.confusion_matrix(prediction,test_list)
print (" Random Forest")
print("accuracy:%.2f%%" %(testscores*100))
print(metrics.classification_report(prediction,test_list,digits=2))
print(confusion)


# VIEW CORPUS OF WORDS
table = str.maketrans('', '', string.punctuation)
blogs_df=pd.read_pickle("blogs_df.pkl")
blog_text=blogs_df['text']
corpus=[]
for text in blog_text:
    words=text.split()
    stripped = [w.translate(table) for w in words]
    corpus.append(stripped)
corpus = [item for sublist in corpus for item in sublist]
data_analysis = nltk.FreqDist(corpus)
data_analysis.plot(25, cumulative=False)

# Create and generate the word cloud:
import matplotlib.pyplot as plt

#convert list to string and generate
unique_string=(" ").join(corpus)
wordcloud = WordCloud(width = 1000, height = 500).generate(unique_string)
plt.figure(figsize=(15,8))
plt.imshow(wordcloud)
plt.axis("off")
#plt.savefig("your_file_name"+".png", bbox_inches='tight')
plt.show()
plt.close()


# This shows all stop words in the top. So we have to remove the stop words
stop = stopwords.words('english')
corpus_no_stop=[]
for text in blog_text:
    words=text.split()
    stripped = [w.translate(table) for w in words]
    words_no_stop=[]
    for word in stripped:
        word=word.lower()
        if word not in stop and ' ' not in word:
            corpus_no_stop.append(word)
data_analysis = nltk.FreqDist(corpus_no_stop)
data_analysis.plot(25, cumulative=False)


# Create and generate the word cloud:
import matplotlib.pyplot as plt

#convert list to string and generate
unique_string=(" ").join(corpus_no_stop)
wordcloud = WordCloud(width = 1000, height = 500).generate(unique_string)
plt.figure(figsize=(15,8))
plt.imshow(wordcloud)
plt.axis("off")
#plt.savefig("your_file_name"+".png", bbox_inches='tight')
plt.show()
plt.close()


# we will remove all stop words from the reviews and run models again
blogs_cleaned=[]
for text in blog_text:
    processed=[]
    words=text.split()
    stripped = [w.translate(table) for w in words]
    for word in stripped:
        word=word.lower()
        if word not in stop and ' ' not in word:
            processed.append(word)
    blogs_cleaned.append(' '.join(processed))


blogs_df['text']=blogs_cleaned


source_male=blogs_df.loc[blogs_df['gender'] == "male"]
source_female=blogs_df.loc[blogs_df['gender'] == "female"]
source_male=source_male.sample(n=31671)
source=pd.concat([source_male,source_female], ignore_index=True)


# USE COUNT VECTORIZER FOR GENDER
print (" USING COUNT VECTORIZER")
train, test = train_test_split(source, test_size=0.20)
vt = CountVectorizer(max_features=100)
vector = vt.fit_transform(train['text'].values).toarray()
test_vector = vt.transform(test['text'].values)
train_gender=train['gender']
test_gender=test['gender']
# multinomial Naive Bayes clasifier
print ("PREDICTIONS FOR GENDER")

clf = MultinomialNB()
model = clf.fit(vector, train_gender.values)
prediction = model.predict(test_vector)
test_list=test_gender.to_list()
testscores = metrics.accuracy_score(prediction,test_list)
confusion = metrics.confusion_matrix(prediction,test_list)
print ("Multinomial Naive Bayes Classifier")
print("accuracy:%.2f%%" %(testscores*100))
print(metrics.classification_report(prediction,test_list,digits=2))
print(confusion)

# Bernoulli clasifier
clf = BernoulliNB()
model = clf.fit(vector, train_gender.values)
prediction = model.predict(test_vector)
test_list=test_gender.to_list()
testscores = metrics.accuracy_score(prediction,test_list)
confusion = metrics.confusion_matrix(prediction,test_list)
print ("Bernoulli Classifier")
print("accuracy:%.2f%%" %(testscores*100))
print(metrics.classification_report(prediction,test_list,digits=2))
print(confusion)

# Linear SVC
clf = LinearSVC()
model = clf.fit(vector, train_gender.values)
prediction = model.predict(test_vector)
test_list=test_gender.to_list()
testscores = metrics.accuracy_score(prediction,test_list)
confusion = metrics.confusion_matrix(prediction,test_list)
print (" Linear Support Vector Classifier")
print("accuracy:%.2f%%" %(testscores*100))
print(metrics.classification_report(prediction,test_list,digits=2))
print(confusion)




# Logistic Regression Clasifier
clf = LogisticRegression()
model = clf.fit(vector, train_gender.values)
prediction = model.predict(test_vector)
test_list=test_gender.to_list()
testscores = metrics.accuracy_score(prediction,test_list)
confusion = metrics.confusion_matrix(prediction,test_list)
print (" Logistic Regression Clasifier")
print("accuracy:%.2f%%" %(testscores*100))
print(metrics.classification_report(prediction,test_list,digits=2))
print(confusion)

# SGD Classifier
clf = SGDClassifier()
model = clf.fit(vector, train_gender.values)
prediction = model.predict(test_vector)
test_list=test_gender.to_list()
testscores = metrics.accuracy_score(prediction,test_list)
confusion = metrics.confusion_matrix(prediction,test_list)
print (" SGD Clasifier")
print("accuracy:%.2f%%" %(testscores*100))
print(metrics.classification_report(prediction,test_list,digits=2))
print(confusion)

# Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100)
model = clf.fit(vector, train_gender.values)
prediction = model.predict(test_vector)
test_list=test_gender.to_list()
testscores = metrics.accuracy_score(prediction,test_list)
confusion = metrics.confusion_matrix(prediction,test_list)
print (" Random Forest")
print("accuracy:%.2f%%" %(testscores*100))
print(metrics.classification_report(prediction,test_list,digits=2))
print(confusion)


# USE COUNT VECTORIZER FOR AGE GROUP

train, test = train_test_split(source, test_size=0.20)
vt = CountVectorizer(max_features=100)
vector = vt.fit_transform(train['text'].values).toarray()
test_vector = vt.transform(test['text'].values)
train_age_group=train['age_group']
test_age_group=test['age_group']

print ("PREDICTIONS FOR AGE")
clf = MultinomialNB()
model = clf.fit(vector, train_age_group.values)
prediction = model.predict(test_vector)
test_list=test_age_group.to_list()
testscores = metrics.accuracy_score(prediction,test_list)
confusion = metrics.confusion_matrix(prediction,test_list)
print ("Multinomial Naive Bayes Classifier")
print("accuracy:%.2f%%" %(testscores*100))
print(metrics.classification_report(prediction,test_list,digits=2))
print(confusion)

# Bernoulli clasifier
clf = BernoulliNB()
model = clf.fit(vector, train_age_group.values)
prediction = model.predict(test_vector)
test_list=test_age_group.to_list()
testscores = metrics.accuracy_score(prediction,test_list)
confusion = metrics.confusion_matrix(prediction,test_list)
print ("Bernoulli Classifier")
print("accuracy:%.2f%%" %(testscores*100))
print(metrics.classification_report(prediction,test_list,digits=2))
print(confusion)

# Linear SVC
clf = LinearSVC()
model = clf.fit(vector, train_age_group.values)
prediction = model.predict(test_vector)
test_list=test_age_group.to_list()
testscores = metrics.accuracy_score(prediction,test_list)
confusion = metrics.confusion_matrix(prediction,test_list)
print (" Linear Support Vector Classifier")
print("accuracy:%.2f%%" %(testscores*100))
print(metrics.classification_report(prediction,test_list,digits=2))
print(confusion)




# Logistic Regression Clasifier
clf = LogisticRegression()
model = clf.fit(vector, train_age_group.values)
prediction = model.predict(test_vector)
test_list=test_age_group.to_list()
testscores = metrics.accuracy_score(prediction,test_list)
confusion = metrics.confusion_matrix(prediction,test_list)
print (" Logistic Regression Clasifier")
print("accuracy:%.2f%%" %(testscores*100))
print(metrics.classification_report(prediction,test_list,digits=2))
print(confusion)

# SGD Classifier
clf = SGDClassifier()
model = clf.fit(vector, train_age_group.values)
prediction = model.predict(test_vector)
test_list=test_age_group.to_list()
testscores = metrics.accuracy_score(prediction,test_list)
confusion = metrics.confusion_matrix(prediction,test_list)
print (" SGD Clasifier")
print("accuracy:%.2f%%" %(testscores*100))
print(metrics.classification_report(prediction,test_list,digits=2))
print(confusion)

# Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100)
model = clf.fit(vector, train_age_group.values)
prediction = model.predict(test_vector)
test_list=test_age_group.to_list()
testscores = metrics.accuracy_score(prediction,test_list)
confusion = metrics.confusion_matrix(prediction,test_list)
print (" Random Forest")
print("accuracy:%.2f%%" %(testscores*100))
print(metrics.classification_report(prediction,test_list,digits=2))
print(confusion)


# USE TF_IDF VECTORIZER FOR GENDER
train, test = train_test_split(source, test_size=0.20)
#vt = CountVectorizer(max_features=100)
vt = TfidfVectorizer(max_features=1000)
vector = vt.fit_transform(train['text'].values).toarray()
test_vector = vt.transform(test['text'].values)
train_gender=train['gender']
test_gender=test['gender']

# multinomial Naive Bayes clasifier
print ("PREDICTIONS FOR GENDER")

clf = MultinomialNB()
model = clf.fit(vector, train_gender.values)
prediction = model.predict(test_vector)
test_list=test_gender.to_list()
testscores = metrics.accuracy_score(prediction,test_list)
confusion = metrics.confusion_matrix(prediction,test_list)
print ("Multinomial Naive Bayes Classifier")
print("accuracy:%.2f%%" %(testscores*100))
print(metrics.classification_report(prediction,test_list,digits=2))
print(confusion)

# Bernoulli clasifier
clf = BernoulliNB()
model = clf.fit(vector, train_gender.values)
prediction = model.predict(test_vector)
test_list=test_gender.to_list()
testscores = metrics.accuracy_score(prediction,test_list)
confusion = metrics.confusion_matrix(prediction,test_list)
print ("Bernoulli Classifier")
print("accuracy:%.2f%%" %(testscores*100))
print(metrics.classification_report(prediction,test_list,digits=2))
print(confusion)

# Linear SVC
clf = LinearSVC()
model = clf.fit(vector, train_gender.values)
prediction = model.predict(test_vector)
test_list=test_gender.to_list()
testscores = metrics.accuracy_score(prediction,test_list)
confusion = metrics.confusion_matrix(prediction,test_list)
print (" Linear Support Vector Classifier")
print("accuracy:%.2f%%" %(testscores*100))
print(metrics.classification_report(prediction,test_list,digits=2))
print(confusion)




# Logistic Regression Clasifier
clf = LogisticRegression()
model = clf.fit(vector, train_gender.values)
prediction = model.predict(test_vector)
test_list=test_gender.to_list()
testscores = metrics.accuracy_score(prediction,test_list)
confusion = metrics.confusion_matrix(prediction,test_list)
print (" Logistic Regression Clasifier")
print("accuracy:%.2f%%" %(testscores*100))
print(metrics.classification_report(prediction,test_list,digits=2))
print(confusion)

# SGD Classifier
clf = SGDClassifier()
model = clf.fit(vector, train_gender.values)
prediction = model.predict(test_vector)
test_list=test_gender.to_list()
testscores = metrics.accuracy_score(prediction,test_list)
confusion = metrics.confusion_matrix(prediction,test_list)
print (" SGD Clasifier")
print("accuracy:%.2f%%" %(testscores*100))
print(metrics.classification_report(prediction,test_list,digits=2))
print(confusion)

# Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100)
model = clf.fit(vector, train_gender.values)
prediction = model.predict(test_vector)
test_list=test_gender.to_list()
testscores = metrics.accuracy_score(prediction,test_list)
confusion = metrics.confusion_matrix(prediction,test_list)
print (" Random Forest")
print("accuracy:%.2f%%" %(testscores*100))
print(metrics.classification_report(prediction,test_list,digits=2))
print(confusion)



print ("PREDICTIONS FOR AGE")
clf = MultinomialNB()
model = clf.fit(vector, train_age_group.values)
prediction = model.predict(test_vector)
test_list=test_age_group.to_list()
testscores = metrics.accuracy_score(prediction,test_list)
confusion = metrics.confusion_matrix(prediction,test_list)
print ("Multinomial Naive Bayes Classifier")
print("accuracy:%.2f%%" %(testscores*100))
print(metrics.classification_report(prediction,test_list,digits=2))
print(confusion)

# Bernoulli clasifier
clf = BernoulliNB()
model = clf.fit(vector, train_age_group.values)
prediction = model.predict(test_vector)
test_list=test_age_group.to_list()
testscores = metrics.accuracy_score(prediction,test_list)
confusion = metrics.confusion_matrix(prediction,test_list)
print ("Bernoulli Classifier")
print("accuracy:%.2f%%" %(testscores*100))
print(metrics.classification_report(prediction,test_list,digits=2))
print(confusion)

# Linear SVC
clf = LinearSVC()
model = clf.fit(vector, train_age_group.values)
prediction = model.predict(test_vector)
test_list=test_age_group.to_list()
testscores = metrics.accuracy_score(prediction,test_list)
confusion = metrics.confusion_matrix(prediction,test_list)
print (" Linear Support Vector Classifier")
print("accuracy:%.2f%%" %(testscores*100))
print(metrics.classification_report(prediction,test_list,digits=2))
print(confusion)




# Logistic Regression Clasifier
clf = LogisticRegression()
model = clf.fit(vector, train_age_group.values)
prediction = model.predict(test_vector)
test_list=test_age_group.to_list()
testscores = metrics.accuracy_score(prediction,test_list)
confusion = metrics.confusion_matrix(prediction,test_list)
print (" Logistic Regression Clasifier")
print("accuracy:%.2f%%" %(testscores*100))
print(metrics.classification_report(prediction,test_list,digits=2))
print(confusion)

# SGD Classifier
clf = SGDClassifier()
model = clf.fit(vector, train_age_group.values)
prediction = model.predict(test_vector)
test_list=test_age_group.to_list()
testscores = metrics.accuracy_score(prediction,test_list)
confusion = metrics.confusion_matrix(prediction,test_list)
print (" SGD Clasifier")
print("accuracy:%.2f%%" %(testscores*100))
print(metrics.classification_report(prediction,test_list,digits=2))
print(confusion)

# Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100)
model = clf.fit(vector, train_age_group.values)
prediction = model.predict(test_vector)
test_list=test_age_group.to_list()
testscores = metrics.accuracy_score(prediction,test_list)
confusion = metrics.confusion_matrix(prediction,test_list)
print (" Random Forest")
print("accuracy:%.2f%%" %(testscores*100))
print(metrics.classification_report(prediction,test_list,digits=2))
print(confusion)




