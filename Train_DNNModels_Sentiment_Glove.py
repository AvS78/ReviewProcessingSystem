
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
import pandas
import os
import tensorflow
from tensorflow.keras.layers import Input
import gensim
from sklearn import metrics


from tensorflow.keras.utils import to_categorical
import os

# Input parameters
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


# generate train and test as sampled
source_positive_sampled_train=source_positive.sample(n=160000)
source_negative_sampled_train=source_negative.sample(n=160000)
source_positive_sampled_test=source_positive[~source_positive.isin(source_positive_sampled_train)].dropna()
source_negative_sampled_test=source_negative[~source_negative.isin(source_negative_sampled_train)].dropna()
y_positive_train=source_positive_sampled_train['sentiment']
y_positive_test=source_positive_sampled_test['sentiment']
y_negative_train=source_negative_sampled_train['sentiment']
y_negative_test=source_negative_sampled_test['sentiment']

x_positive_train=source_positive_sampled_train['processed_with_lemmatize']
x_positive_test=source_positive_sampled_test['processed_with_lemmatize']
x_negative_train=source_negative_sampled_train['processed_with_lemmatize']
x_negative_test=source_negative_sampled_test['processed_with_lemmatize']

x_train=pd.concat([x_positive_train,x_negative_train], ignore_index=True)
x_test=pd.concat([x_positive_test,x_negative_test], ignore_index=True)

y_train=pd.concat([y_positive_train,y_negative_train], ignore_index=True)
y_test=pd.concat([y_positive_test,y_negative_test], ignore_index=True)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)


y_cat_train=to_categorical(y_train)
y_cat_test=to_categorical(y_test)
x=x_train
y=y_cat_train


#use glove model for embedding
word_2_vec_glove_path="glove.6B.300d.txt"
glove_embeddings_dict = {}
with open(word_2_vec_glove_path, 'r', encoding="utf-8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], "float32")
        glove_embeddings_dict[word] = vector


#remove words that are not in the glove embeddings from the review text
def remove_unknown(string):
    words=string.split(" ")
    for word in words:
        if word not in glove_embeddings_dict.keys():
            words.remove(word)
    updated=" ".join(words)
    if(string != updated):
        print ("original review ****", string)
        print("updated review****", updated)
    return (updated)

x_updated=[]
for review in x:
    updated=remove_unknown(review)
    x_updated.append(updated)

x_test_updated=[]
for review in x_test:
    updated=remove_unknown(review)
    x_test_updated.append(updated)


x=pd.Series(x_updated)
x_test=pd.Series(x_test_updated)


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


#embedding_layer output is a 2 D vector
embedding_layer = Embedding(len(word_index)+1, #8820 - the corpus vocabulary size
                            embedding_size, # 300 - embedding dimensions
                            weights=[embedding_matrix], #weights from the embedding matrix - learnt from google pre-trained model
                            input_length=max_len) #200: padded review size 

# Simple RNN
no_of_epochs=2
model = Sequential()
model.add(embedding_layer)
model.add(SimpleRNN(output_size, activation=rnn_activation))
model.add(Dropout(0.25))
model.add(Dense(128))
model.add(Dense(2))
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
#          callbacks=[checkpoint, csv_logger])


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



# GRU

model = Sequential()
model.add(embedding_layer)
model.add(GRU(output_size,activation=rnn_activation,recurrent_activation=recurrent_activation))
model.add(Dropout(0.25))
model.add(Dense(2))
model.add(Activation('sigmoid'))
model.summary()



model.compile(loss=loss,
              optimizer=optimizer,
              metrics=['accuracy'])


print('GRU')
nb_epoch=2
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
no_of_epochs=1
model = Sequential()
model.add(embedding_layer)
model.add(Bidirectional(LSTM(output_size,activation=rnn_activation,recurrent_activation=recurrent_activation)))
model.add(Dropout(0.25))
model.add(Dense(2))
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
model.add(Dense(2))
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
model.add(Dense(2))
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
model.add(Dense(2))
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


# Regular CNN 

model = Sequential()
model.add(embedding_layer)
model.add(Dropout(0.25))
model.add(Convolution1D(nb_filter,filter_length, activation="relu", padding='same'))
model.add(MaxPooling1D(5))
model.add(Convolution1D(64, 5, activation=cnn_activation, padding='same'))
model.add(MaxPooling1D(pool_length))
model.add(Flatten())
model.add(Dense(64, activation="relu"))
model.add(Dense(2, activation="sigmoid"))
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
model.add(Dense(2))
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



