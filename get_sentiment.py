

import pandas as pd
import pickle
import pandas

from sklearn import metrics
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import re
import nltk
from nltk.corpus import stopwords
stop = stopwords.words('english')
wnl=nltk.WordNetLemmatizer()

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



def return_sentiment(text):
    test_review=clean_text(text)
    processed=pd.read_pickle(r"F:\NUS-NLP Project\sentiment_dataframe.pkl")
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
    check_vector=tfidfX[415918]
    tfidfX=tfidfX[:415918]
    tfidfX=tfidfX[:415918]
    x_train, x_test, y_train, y_test=train_test_split(tfidfX, source.sentiment,test_size=0.2, random_state=42, shuffle=True,stratify=source['sentiment'])
    model = mln.fit(x_train, y_train)
    result=model.predict(check_vector)
    result=result[0]
    if result==1:
        result='happy'
    else:
        result='unhappy'
    return(result)



# print(return_sentiment(text))

# processed=pd.read_pickle(r"C:\Users\rbfor\OneDrive\Documents\NLP_Project\sentiment_dataframe.pkl")
# source=processed
# vect = CountVectorizer()
# tfidf = TfidfTransformer()
# mln = MultinomialNB()
# vectorizer = CountVectorizer(max_features=100000, min_df=5, max_df=0.7)
# X=source['processed_with_lemmatize']
# y=X.tolist()
# y.append(test_review)
# X=pd.Series(y)
# vX = vect.fit_transform(X)
# tfidfX = tfidf.fit_transform(vX)
# check_vector=tfidfX[415918]
# tfidfX=tfidfX[:415918]
# tfidfX=tfidfX[:415918]
# x_train, x_test, y_train, y_test=train_test_split(tfidfX, source.sentiment,test_size=0.2, random_state=42, shuffle=True,stratify=source['sentiment'])
# model = mln.fit(x_train, y_train)
# result=model.predict(check_vector)
# result=result[0]
# if result==1:
#     result='happy'
# else:
#     result='unhappy'



# print(result)







# # In[39]:


# vect = CountVectorizer()
# tfidf = TfidfTransformer()
# clf = MultinomialNB()
# vectorizer = CountVectorizer(max_features=100000, min_df=5, max_df=0.7)
# X=source['processed_with_pos']
# vX = vect.fit_transform(X)
# tfidfX = tfidf.fit_transform(vX)


# # In[40]:


# x_train, x_test, y_train, y_test=train_test_split(tfidfX, source.sentiment,test_size=0.2, random_state=42, shuffle=True,stratify=source['sentiment'])


# # In[41]:


# model = clf.fit(x_train, y_train)
# y_pred=model.predict(x_test)
# testscores = metrics.accuracy_score(y_test,y_pred)
# confusion = metrics.confusion_matrix(y_test,y_pred)
# print("accuracy:%.2f%%" %(testscores*100))
# print(metrics.classification_report(y_test,y_pred,digits=2))
# print(confusion)


# # In[85]:


# from sklearn import tree
# dt=tree.DecisionTreeClassifier()


# # In[86]:


# model = dt.fit(x_train, y_train)
# y_pred=model.predict(x_test)
# testscores = metrics.accuracy_score(y_test,y_pred)
# confusion = metrics.confusion_matrix(y_test,y_pred)
# print("accuracy:%.2f%%" %(testscores*100))
# print(metrics.classification_report(y_test,y_pred,digits=2))
# print(confusion)


# # In[97]:


# from sklearn.linear_model import SGDClassifier
# from sklearn.svm import SVC


# # In[102]:


# sgc=SGDClassifier()
# model = vcf.fit(x_train, y_train)
# y_pred=model.predict(x_test)
# testscores = metrics.accuracy_score(y_test,y_pred)
# confusion = metrics.confusion_matrix(y_test,y_pred)
# print("accuracy:%.2f%%" %(testscores*100))
# print(metrics.classification_report(y_test,y_pred,digits=2))
# print(confusion)


# # In[101]:


# vcf=sklearn.svm.LinearSVC(C=1.0)
# model = vcf.fit(x_train, y_train)
# y_pred=model.predict(x_test)
# testscores = metrics.accuracy_score(y_test,y_pred)
# confusion = metrics.confusion_matrix(y_test,y_pred)
# print("accuracy:%.2f%%" %(testscores*100))
# print(metrics.classification_report(y_test,y_pred,digits=2))
# print(confusion)


# # In[ ]:





# # In[103]:


# x_train, x_test, y_train, y_test=train_test_split(source.processed_with_lemmatize, source.sentiment,test_size=0.2, random_state=42, shuffle=True,stratify=source['sentiment'])


# # In[ ]:


# tokenizer=Tokenizer(num)


# # In[ ]:





# # In[ ]:





# # In[95]:


# x=x_train.toarray()


# # In[88]:


# from tensorflow.keras.models import Sequential
# from tensorflow.keras import layers

# input_dim = x_train.shape[1]  # Number of features


# # In[90]:


# model = Sequential()
# model.add(layers.Dense(10, input_dim=input_dim, activation='relu'))
# model.add(layers.Dense(1, activation='sigmoid'))
# model.compile(loss='binary_crossentropy',  optimizer='adam', metrics=['accuracy'])
# model.summary()


# # In[1]:


# history = model.fit(x_train, y_train,
#                     epochs=10,
#                     verbose=True,
#                     )


# # In[6]:


# # use only noun words in vocabulary
# import spacy
# import en_core_web_sm
# nlp = en_core_web_sm.load()
# doc = nlp(a[0])
# a=source['processed_with_lemmatize']
# # Token and Tag 
# for token in doc:
#     print(token, token.pos_)


# # In[20]:





# # In[21]:


# a[0]


# # In[23]:





# # In[ ]:




