
import pandas as pd
from collections import Counter
from nltk import FreqDist
import pickle
import sys
import numpy as np
import numpy
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import LSTM,Bidirectional
from tensorflow.keras.layers import SimpleRNN
from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import Convolution1D, MaxPooling1D, Concatenate
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing import text,sequence
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from tensorflow.keras.utils import to_categorical
import pandas
import os
import tensorflow
from tensorflow.keras.layers import Input
from gensim.models import Word2Vec
from sklearn import metrics
import gensim
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
import os

max_features = 10000
max_len = 50
embedding_size = 100

# Convolution parameters
filter_length = 3
nb_filter = 150
pool_length = 2
cnn_activation = 'relu'

# RNN parameters
output_size = 50
rnn_activation = 'tanh'
recurrent_activation = 'hard_sigmoid'

# Compile parameters
loss = 'binary_crossentropy'
optimizer = 'rmsprop'

# Training parameters
batch_size = 50
validation_split = 0.25
shuffle = True

processed=pd.read_pickle("/content/drive/My Drive/combined_balanced_100k_text_processed_2610.pkl")

train, test=train_test_split(processed, test_size=0.2, shuffle=True, random_state=42, stratify=processed['dept'])
train.groupby(['dept']).agg(['count'])

x_train=train['processed_with_lemmatize']
x_test=test['processed_with_lemmatize']
y_train=train['dept']
y_test=test['dept']

y_train=y_train.replace('beauty', 0)
y_train=y_train.replace('cell', 1)
y_train=y_train.replace('clothing', 2)
y_train=y_train.replace('electronics', 3)
y_train=y_train.replace('grocery', 4)
y_train=y_train.replace('health', 5)
y_train=y_train.replace('home', 6)
y_train=y_train.replace('movies', 7)
y_train=y_train.replace('pet', 8)
y_train=y_train.replace('toys', 9)

y_test=y_test.replace('beauty', 0)
y_test=y_test.replace('cell', 1)
y_test=y_test.replace('clothing', 2)
y_test=y_test.replace('electronics', 3)
y_test=y_test.replace('grocery', 4)
y_test=y_test.replace('health', 5)
y_test=y_test.replace('home', 6)
y_test=y_test.replace('movies', 7)
y_test=y_test.replace('pet', 8)
y_test=y_test.replace('toys', 9)

y_cat_train=to_categorical(y_train)
y_cat_test=to_categorical(y_test)

x=x_train
y=y_cat_train


#get the own pre-trained word2vec model
w2v = Word2Vec.load('/content/drive/My Drive/created_model_100k.bin')

from tensorflow.keras.preprocessing import text
tk=tensorflow.keras.preprocessing.text.Tokenizer(
    num_words=max_features, 
    filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
    lower=True,
    split=" "
)
tk.fit_on_texts(x)
x_seq = tk.texts_to_sequences(x)

word_index = tk.word_index #creates dictionary for the input tokenized corpus x
x_seq_pad = sequence.pad_sequences(x_seq,maxlen=max_len) #pads to max length = 50
embedding_matrix = numpy.zeros((len(word_index) + 1, embedding_size))
for word,i in word_index.items():
    #if word in w2v.vocab:
    if word in w2v.wv.vocab.keys():
    #if word in glove_embeddings_dict.keys():
        embedding_matrix[i] = w2v.wv[word]
        #embedding_matrix[i]=glove_embeddings_dict[word]
#embedding_layer output is a 2 D vector
embedding_layer = Embedding(len(word_index)+1, #8820 - the corpus vocabulary size
                            embedding_size, # 300 - embedding dimensions
                            weights=[embedding_matrix], #weights from the embedding matrix - learnt from google pre-trained model
                            input_length=max_len) #200: padded review size


no_of_epochs=5

model = Sequential()
model.add(embedding_layer)# 200x300 produces for each of the padded review (200 tokens), with each token of 300 dimensions per google pretrained word embedding
model.add(SimpleRNN(output_size, activation=rnn_activation))
model.add(Dropout(0.25))
model.add(Dense(128))
model.add(Dense(10))
model.add(Activation('sigmoid'))
model.compile(loss=loss,
              optimizer=optimizer,
              metrics=['accuracy'])
model.summary()

print('Simple RNN')
model.fit(x_seq_pad, 
          y, 
          batch_size=batch_size, 
          epochs=no_of_epochs,
          validation_split=validation_split,
          shuffle=True)

x_test
tk.fit_on_texts(x_test)
x_test_seq = tk.texts_to_sequences(x_test)
x_test_seq_pad = sequence.pad_sequences(x_test_seq, maxlen=50)
ypred=model.predict(x_test_seq_pad)
ypredout= np.argmax(ypred,axis=1)
testscores = metrics.accuracy_score(y_test,ypredout)
confusion = metrics.confusion_matrix(y_test,ypredout)
print("accuracy:%.2f%%" %(testscores*100))
print(metrics.classification_report(y_test,ypredout,digits=2))
print(confusion)

no_of_epochs=5
model = Sequential()
model.add(embedding_layer)
model.add(GRU(output_size,activation=rnn_activation,recurrent_activation=recurrent_activation))
model.add(Dropout(0.25))
model.add(Dense(10))
model.add(Activation('sigmoid'))
model.summary()

model.compile(loss=loss,
              optimizer=optimizer,
              metrics=['accuracy'])

print('GRU')
#model.fit(x, y, batch_size=batch_size, nb_epoch=nb_epoch,validation_split=validation_split,shuffle=shuffle)
model.fit(x_seq_pad, y, batch_size=batch_size, epochs=no_of_epochs,validation_split=validation_split,shuffle=shuffle)

x_test
tk.fit_on_texts(x_test)
x_test_seq = tk.texts_to_sequences(x_test)
x_test_seq_pad = sequence.pad_sequences(x_test_seq, maxlen=50)
ypred=model.predict(x_test_seq_pad)
ypredout= np.argmax(ypred,axis=1)
testscores = metrics.accuracy_score(y_test,ypredout)
confusion = metrics.confusion_matrix(y_test,ypredout)
print("accuracy:%.2f%%" %(testscores*100))
print(metrics.classification_report(y_test,ypredout,digits=2))
print(confusion)

# Bidirectional LSTM
no_of_epochs=5
model = Sequential()
model.add(embedding_layer)
model.add(Bidirectional(LSTM(output_size,activation=rnn_activation,recurrent_activation=recurrent_activation)))
model.add(Dropout(0.25))
model.add(Dense(10))
model.add(Activation('sigmoid'))
model.summary()

model.compile(loss=loss,optimizer=optimizer,metrics=['accuracy'])
print('Bidirectional LSTM')
model.fit(x_seq_pad, y, batch_size=batch_size, epochs=no_of_epochs,validation_split=validation_split,shuffle=shuffle)

x_test
tk.fit_on_texts(x_test)
x_test_seq = tk.texts_to_sequences(x_test)
x_test_seq_pad = sequence.pad_sequences(x_test_seq, maxlen=50)
ypred=model.predict(x_test_seq_pad)
ypredout= np.argmax(ypred,axis=1)
testscores = metrics.accuracy_score(y_test,ypredout)
confusion = metrics.confusion_matrix(y_test,ypredout)
print("accuracy:%.2f%%" %(testscores*100))
print(metrics.classification_report(y_test,ypredout,digits=2))
print(confusion)

# LSTM
model = Sequential()
model.add(embedding_layer)
model.add(Dropout(0.25))
model.add(LSTM(output_size,activation=rnn_activation,recurrent_activation=recurrent_activation))
model.add(Dropout(0.25))
model.add(Dense(10))
model.add(Activation('sigmoid'))
model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
model.summary()

print('LSTM')
model.fit(x_seq_pad, y, batch_size=batch_size, epochs=no_of_epochs,validation_split=validation_split,shuffle=shuffle)

x_test
tk.fit_on_texts(x_test)
x_test_seq = tk.texts_to_sequences(x_test)
x_test_seq_pad = sequence.pad_sequences(x_test_seq, maxlen=50)
ypred=model.predict(x_test_seq_pad)
ypredout= np.argmax(ypred,axis=1)
testscores = metrics.accuracy_score(y_test,ypredout)
confusion = metrics.confusion_matrix(y_test,ypredout)
print("accuracy:%.2f%%" %(testscores*100))
print(metrics.classification_report(y_test,ypredout,digits=2))
print(confusion)

# CNN + LSTM

model = Sequential()
model.add(embedding_layer)
model.add(Dropout(0.5))
model.add(Convolution1D(nb_filter,
                        filter_length,
                        padding='valid',
                        activation=cnn_activation
                        ))
model.add(MaxPooling1D(pool_length))
model.add(LSTM(output_size,activation=rnn_activation,recurrent_activation=recurrent_activation))
model.add(Dropout(0.25))
model.add(Dense(10))
model.add(Activation('sigmoid'))
model.compile(loss=loss,
              optimizer=optimizer,
              metrics=['accuracy'])
model.summary()

print('CNN + LSTM')
model.fit(x_seq_pad, y, batch_size=batch_size, epochs=no_of_epochs,validation_split=validation_split,shuffle=shuffle)

x_test
tk.fit_on_texts(x_test)
x_test_seq = tk.texts_to_sequences(x_test)
x_test_seq_pad = sequence.pad_sequences(x_test_seq, maxlen=50)
ypred=model.predict(x_test_seq_pad)
ypredout= np.argmax(ypred,axis=1)
testscores = metrics.accuracy_score(y_test,ypredout)
confusion = metrics.confusion_matrix(y_test,ypredout)
print("accuracy:%.2f%%" %(testscores*100))
print(metrics.classification_report(y_test,ypredout,digits=2))
print(confusion)

# CNN
filter_sizes = (3,5,7,9)
num_filters = 100
graph_in = Input(shape=(max_len, embedding_size))
convs = []
for fsz in filter_sizes:
    conv = Convolution1D(num_filters, fsz,padding='valid',activation='relu')(graph_in)
    pool = MaxPooling1D(2)(conv)
    flatten = Flatten()(pool)
    convs.append(flatten)

if len(filter_sizes) > 1:
    out = Concatenate()(convs)
else:
    out = convs[0]

graph = Model(graph_in,out)
model = Sequential()
model.add(embedding_layer)
model.add(Dropout(0.25, input_shape=(max_len, embedding_size)))
model.add(graph)
model.add(Dense(64))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('sigmoid'))
model.summary()

model.compile(loss=loss, optimizer='rmsprop', metrics=['accuracy'])
model.fit(x_seq_pad, y, batch_size=batch_size, epochs=no_of_epochs,validation_split=validation_split,shuffle=shuffle)

x_test
tk.fit_on_texts(x_test)
x_test_seq = tk.texts_to_sequences(x_test)
x_test_seq_pad = sequence.pad_sequences(x_test_seq, maxlen=50)
ypred=model.predict(x_test_seq_pad)
ypredout= np.argmax(ypred,axis=1)
testscores = metrics.accuracy_score(y_test,ypredout)
confusion = metrics.confusion_matrix(y_test,ypredout)
print("accuracy:%.2f%%" %(testscores*100))
print(metrics.classification_report(y_test,ypredout,digits=2))
print(confusion)

opt = SGD(lr=0.01, momentum=0.80, decay=1e-6, nesterov=True)
model.compile(loss=loss, optimizer=opt, metrics=['accuracy'])
model.fit(x_seq_pad, y, batch_size=batch_size, epochs=no_of_epochs,validation_split=validation_split,shuffle=shuffle)

x_test
tk.fit_on_texts(x_test)
x_test_seq = tk.texts_to_sequences(x_test)
x_test_seq_pad = sequence.pad_sequences(x_test_seq, maxlen=50)
ypred=model.predict(x_test_seq_pad)
ypredout= np.argmax(ypred,axis=1)
testscores = metrics.accuracy_score(y_test,ypredout)
confusion = metrics.confusion_matrix(y_test,ypredout)
print("accuracy:%.2f%%" %(testscores*100))
print(metrics.classification_report(y_test,ypredout,digits=2))
print(confusion)

model = Sequential()
model.add(embedding_layer)
model.add(Dropout(0.25))
model.add(Convolution1D(nb_filter,filter_length, activation="relu", padding='same'))
model.add(MaxPooling1D(5))
model.add(Convolution1D(64, 5, activation=cnn_activation, padding='same'))
model.add(MaxPooling1D(pool_length))
model.add(Flatten())
model.add(Dense(64, activation="relu"))
model.add(Dense(10, activation="sigmoid"))
model.summary()

model.compile(loss=loss,optimizer=optimizer, metrics=['accuracy'])


print('CNN')
model.fit(x_seq_pad, y, batch_size=batch_size, epochs=no_of_epochs,validation_split=validation_split,shuffle=shuffle)

x_test
tk.fit_on_texts(x_test)
x_test_seq = tk.texts_to_sequences(x_test)
x_test_seq_pad = sequence.pad_sequences(x_test_seq, maxlen=50)
ypred=model.predict(x_test_seq_pad)
ypredout= np.argmax(ypred,axis=1)
testscores = metrics.accuracy_score(y_test,ypredout)
confusion = metrics.confusion_matrix(y_test,ypredout)
print("accuracy:%.2f%%" %(testscores*100))
print(metrics.classification_report(y_test,ypredout,digits=2))
print(confusion)

#DNN

model = Sequential()
model.add(embedding_layer)
model.add(Dense(300))
model.add(Dense(300))
model.add(Dense(300))

model.add(Dropout(0.25, input_shape=(max_len, embedding_size)))
model.add(Flatten())

model.add(Dense(64))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('sigmoid'))
model.summary()

model.compile(loss=loss,optimizer=optimizer, metrics=['accuracy'])
print('CNN')
model.fit(x_seq_pad, y, batch_size=batch_size, epochs=1,validation_split=validation_split,shuffle=shuffle)

x_test
tk.fit_on_texts(x_test)
x_test_seq = tk.texts_to_sequences(x_test)
x_test_seq_pad = sequence.pad_sequences(x_test_seq, maxlen=50)
ypred=model.predict(x_test_seq_pad)
ypredout= np.argmax(ypred,axis=1)
testscores = metrics.accuracy_score(y_test,ypredout)
confusion = metrics.confusion_matrix(y_test,ypredout)
print("accuracy:%.2f%%" %(testscores*100))
print(metrics.classification_report(y_test,ypredout,digits=2))
print(confusion)