# -*- coding: utf-8 -*-

__author__ = 'Oswaldo Ludwig'
__version__ = '1.01'

from keras.layers import Input, Embedding, LSTM, Dense, RepeatVector, Bidirectional, Dropout, merge
from keras.optimizers import Adam, SGD
from keras.models import Model
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.callbacks import EarlyStopping
from keras.preprocessing import sequence

import keras.backend as K
import numpy as np
np.random.seed(1234)  # for reproducibility
import cPickle
import theano.tensor as T
import os
import pandas as pd
import sys
import matplotlib.pyplot as plt

word_embedding_size = 100
sentence_embedding_size = 300
dictionary_size = 7000
maxlen_input = 50
maxlen_output = 50
num_subsets = 1
Epochs = 100
BatchSize = 128  #  Check the capacity of your GPU
Patience = 0
dropout = .25
n_test = 100

vocabulary_file = 'vocabulary_movie'
questions_file = 'Padded_context'
answers_file = 'Padded_answers'
weights_file = 'my_model_weights20.h5'
GLOVE_DIR = './glove.6B/'

early_stopping = EarlyStopping(monitor='val_loss', patience=Patience)



def print_result(input):

    ans_partial = np.zeros((1,maxlen_input))
    ans_partial[0, -1] = 2  #  the index of the symbol BOS (begin of sentence)
    for k in range(maxlen_input - 1):
        ye = model.predict([input, ans_partial])
        mp = np.argmax(ye)
        ans_partial[0, 0:-1] = ans_partial[0, 1:]
        ans_partial[0, -1] = mp
    text = ''
    for k in ans_partial[0]:
        k = k.astype(int)
        if k < (dictionary_size-2):
            w = vocabulary[k]
            text = text + w[0] + ' '
    return(text)


# **********************************************************************
# Reading a pre-trained word embedding and addapting to our vocabulary:
# **********************************************************************

embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))
embedding_matrix = np.zeros((dictionary_size, word_embedding_size))

# Loading our vocabulary:
vocabulary = cPickle.load(open(vocabulary_file, 'rb'))

# Using the Glove embedding:
i = 0
for word in vocabulary:
    embedding_vector = embeddings_index.get(word[0])
    
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
    i += 1

# *******************************************************************
# Keras model of the chatbot: 
# *******************************************************************

ad = Adam(lr=0.00005) 

input_context = Input(shape=(maxlen_input,), dtype='int32', name='input_context')
input_answer = Input(shape=(maxlen_input,), dtype='int32', name='input_answer')
LSTM_encoder = LSTM(sentence_embedding_size, init= 'lecun_uniform')
LSTM_decoder = LSTM(sentence_embedding_size, init= 'lecun_uniform')
if os.path.isfile(weights_file):
    Shared_Embedding = Embedding(output_dim=word_embedding_size, input_dim=dictionary_size, input_length=maxlen_input)
else:
    Shared_Embedding = Embedding(output_dim=word_embedding_size, input_dim=dictionary_size, weights=[embedding_matrix], input_length=maxlen_input)
word_embedding_context = Shared_Embedding(input_context)
context_embedding = LSTM_encoder(word_embedding_context)

word_embedding_answer = Shared_Embedding(input_answer)
answer_embedding = LSTM_decoder(word_embedding_answer)

merge_layer = merge([context_embedding, answer_embedding], mode='concat', concat_axis=1)
out = Dense(dictionary_size/2, activation="relu")(merge_layer)
out = Dense(dictionary_size, activation="softmax")(out)

model = Model(input=[input_context, input_answer], output = [out])

model.compile(loss='categorical_crossentropy', optimizer=ad)

if os.path.isfile(weights_file):
    model.load_weights(weights_file)

# ************************************************************************
# Loading the data:
# ************************************************************************

q = cPickle.load(open(questions_file, 'rb'))
a = cPickle.load(open(answers_file, 'rb'))
n_exem, n_words = a.shape

qt = q[0:n_test,:]
at = a[0:n_test,:]
q = q[n_test + 1:,:]
a = a[n_test + 1:,:]

print('Number of exemples = %d'%(n_exem - n_test))
step = np.around((n_exem - n_test)/num_subsets)
round_exem = step * num_subsets

# *************************************************************************
# Bot training:
# *************************************************************************

x = range(0,Epochs) 
valid_loss = np.zeros(Epochs)
train_loss = np.zeros(Epochs)
for m in range(Epochs):
    
    # Loop over training batches due to memory constraints:
    for n in range(0,round_exem,step):
        
        q2 = q[n:n+step]
        s = q2.shape
        count = 0
        for i, sent in enumerate(a[n:n+step]):
            l = np.where(sent==3)  #  the position od the symbol EOS
            limit = l[0][0]
            count += limit + 1
            
        Q = np.zeros((count,maxlen_input))
        A = np.zeros((count,maxlen_input))
        Y = np.zeros((count,dictionary_size))
        
        # Loop over the training examples:
        count = 0
        for i, sent in enumerate(a[n:n+step]):
            ans_partial = np.zeros((1,maxlen_input))
            
            # Loop over the positions of the current target output (the current output sequence):
            l = np.where(sent==3)  #  the position of the symbol EOS
            limit = l[0][0]

            for k in range(1,limit+1):
                # Mapping the target output (the next output word) for one-hot codding:
                y = np.zeros((1, dictionary_size))
                y[0, sent[k]] = 1

                # preparing the partial answer to input:

                ans_partial[0,-k:] = sent[0:k]

                # training the model for one epoch using teacher forcing:
                
                Q[count, :] = q2[i:i+1] 
                A[count, :] = ans_partial 
                Y[count, :] = y
                count += 1
                
        print('Training epoch: %d, training examples: %d - %d'%(m,n, n + step))
        model.fit([Q, A], Y, batch_size=BatchSize, epochs=1)
         
        test_input = qt[41:42]
        print(print_result(test_input))
        train_input = q[41:42]
        print(print_result(train_input))        
        
    model.save_weights(weights_file, overwrite=True)
