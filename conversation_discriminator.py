# -*- coding: utf-8 -*-

__author__ = 'Oswaldo Ludwig'
__version__ = '1.01'

from keras.layers import Input, Embedding, LSTM, Dense, RepeatVector, Dropout, merge
from keras.optimizers import Adam 
from keras.models import Model
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.preprocessing import sequence
from keras.layers import concatenate

import keras.backend as K
import numpy as np
np.random.seed(1234)  # for reproducibility
import cPickle
import theano
import os.path
import sys
import nltk
import re
import time

from keras.utils import plot_model

word_embedding_size = 100
sentence_embedding_size = 300
dictionary_size = 7000
maxlen_input = 50
learning_rate = 0.000001

vocabulary_file = 'vocabulary_movie'
weights_file = 'my_model_weights20.h5'
weights_file_GAN = 'my_model_weights.h5'
weights_file_discrim = 'my_model_weights_discriminator.h5'
unknown_token = 'something'
file_saved_context = 'saved_context'
file_saved_answer = 'saved_answer'
name_of_computer = 'john'

def greedy_decoder(input):

    flag = 0
    prob = 1
    ans_partial = np.zeros((1,maxlen_input))
    ans_partial[0, -1] = 2  #  the index of the symbol BOS (begin of sentence)
    for k in range(maxlen_input - 1):
        ye = model.predict([input, ans_partial])
        yel = ye[0,:]
        p = np.max(yel)
        mp = np.argmax(ye)
        ans_partial[0, 0:-1] = ans_partial[0, 1:]
        ans_partial[0, -1] = mp
        if mp == 3:  #  he index of the symbol EOS (end of sentence)
            flag = 1
        if flag == 0:    
            prob = prob * p
    text = ''
    for k in ans_partial[0]:
        k = k.astype(int)
        if k < (dictionary_size-2):
            w = vocabulary[k]
            text = text + w[0] + ' '
    return(text, prob)
    
    
def preprocess(raw_word, name):
    
    l1 = ['won’t','won\'t','wouldn’t','wouldn\'t','’m', '’re', '’ve', '’ll', '’s','’d', 'n’t', '\'m', '\'re', '\'ve', '\'ll', '\'s', '\'d', 'can\'t', 'n\'t', 'B: ', 'A: ', ',', ';', '.', '?', '!', ':', '. ?', ',   .', '. ,', 'EOS', 'BOS', 'eos', 'bos']
    l2 = ['will not','will not','would not','would not',' am', ' are', ' have', ' will', ' is', ' had', ' not', ' am', ' are', ' have', ' will', ' is', ' had', 'can not', ' not', '', '', ' ,', ' ;', ' .', ' ?', ' !', ' :', '? ', '.', ',', '', '', '', '']
    l3 = ['-', '_', ' *', ' /', '* ', '/ ', '\"', ' \\"', '\\ ', '--', '...', '. . .']
    l4 = ['jeffrey','fred','benjamin','paula','walter','rachel','andy','helen','harrington','kathy','ronnie','carl','annie','cole','ike','milo','cole','rick','johnny','loretta','cornelius','claire','romeo','casey','johnson','rudy','stanzi','cosgrove','wolfi','kevin','paulie','cindy','paulie','enzo','mikey','i\97','davis','jeffrey','norman','johnson','dolores','tom','brian','bruce','john','laurie','stella','dignan','elaine','jack','christ','george','frank','mary','amon','david','tom','joe','paul','sam','charlie','bob','marry','walter','james','jimmy','michael','rose','jim','peter','nick','eddie','johnny','jake','ted','mike','billy','louis','ed','jerry','alex','charles','tommy','bobby','betty','sid','dave','jeffrey','jeff','marty','richard','otis','gale','fred','bill','jones','smith','mickey']    

    raw_word = raw_word.lower()
    raw_word = raw_word.replace(', ' + name_of_computer, '')
    raw_word = raw_word.replace(name_of_computer + ' ,', '')

    for j, term in enumerate(l1):
        raw_word = raw_word.replace(term,l2[j])
        
    for term in l3:
        raw_word = raw_word.replace(term,' ')
    
    for term in l4:
        raw_word = raw_word.replace(', ' + term, ', ' + name)
        raw_word = raw_word.replace(' ' + term + ' ,' ,' ' + name + ' ,')
        raw_word = raw_word.replace('i am ' + term, 'i am ' + name_of_computer)
        raw_word = raw_word.replace('my name is' + term, 'my name is ' + name_of_computer)
    
    for j in range(30):
        raw_word = raw_word.replace('. .', '')
        raw_word = raw_word.replace('.  .', '')
        raw_word = raw_word.replace('..', '')
       
    for j in range(5):
        raw_word = raw_word.replace('  ', ' ')
        
    if raw_word[-1] <>  '!' and raw_word[-1] <> '?' and raw_word[-1] <> '.' and raw_word[-2:] <>  '! ' and raw_word[-2:] <> '? ' and raw_word[-2:] <> '. ':
        raw_word = raw_word + ' .'
    
    if raw_word == ' !' or raw_word == ' ?' or raw_word == ' .' or raw_word == ' ! ' or raw_word == ' ? ' or raw_word == ' . ':
        raw_word = 'what ?'
    
    if raw_word == '  .' or raw_word == ' .' or raw_word == '  . ':
        raw_word = 'i do not want to talk about it .'
      
    return raw_word

def tokenize(sentences):

    # Tokenizing the sentences into words:
    tokenized_sentences = nltk.word_tokenize(sentences.decode('utf-8'))
    index_to_word = [x[0] for x in vocabulary]
    word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])
    tokenized_sentences = [w if w in word_to_index else unknown_token for w in tokenized_sentences]
    X = np.asarray([word_to_index[w] for w in tokenized_sentences])
    s = X.size
    Q = np.zeros((1,maxlen_input))
    if s < (maxlen_input + 1):
        Q[0,- s:] = X
    else:
        Q[0,:] = X[- maxlen_input:]
    
    return Q

# Open files to save the conversation for further training:
qf = open(file_saved_context, 'w')
af = open(file_saved_answer, 'w')


def init_model():

    # *******************************************************************
    # Keras model of the discriminator: 
    # *******************************************************************

    ad = Adam(lr=learning_rate) 

    input_context = Input(shape=(maxlen_input,), dtype='int32', name='input context')
    input_answer = Input(shape=(maxlen_input,), dtype='int32', name='input answer')
    input_current_token = Input(shape=(dictionary_size,), name='input_current_token')

    LSTM_encoder_discriminator = LSTM(sentence_embedding_size, kernel_initializer= 'lecun_uniform', name = 'encoder discriminator')
    LSTM_decoder_discriminator = LSTM(sentence_embedding_size, kernel_initializer= 'lecun_uniform', name = 'decoder discriminator')
    
    Shared_Embedding = Embedding(output_dim=word_embedding_size, input_dim=dictionary_size, input_length=maxlen_input, trainable=False, name = 'shared')
    word_embedding_context = Shared_Embedding(input_context)
    word_embedding_answer = Shared_Embedding(input_answer)
    context_embedding_discriminator = LSTM_encoder_discriminator(word_embedding_context)
    answer_embedding_discriminator = LSTM_decoder_discriminator(word_embedding_answer)
    loss = concatenate([context_embedding_discriminator, answer_embedding_discriminator, input_current_token], axis=1, name = 'concatenation discriminator')
    loss = Dense(1, activation="sigmoid", name = 'discriminator output')(loss)

    model_discrim = Model(inputs=[input_context, input_answer, input_current_token], outputs = [loss])

    model_discrim.compile(loss='binary_crossentropy', optimizer=ad)

    if os.path.isfile(weights_file_discrim):
        model_discrim.load_weights(weights_file_discrim)
        
    return model_discrim    

def run_discriminator(q, a):

    sa = (a != 0).sum()

    # *************************************************************************
    # running discriminator:
    # *************************************************************************

    p = 1
    m = 0
    model_discrim = init_model()
    count = 0
 
    for i, sent in enumerate(a):
        l = np.where(sent==3)  #  the position od the symbol EOS
        limit = l[0][0]
        count += limit + 1

    Q = np.zeros((count,maxlen_input))
    A = np.zeros((count,maxlen_input))
    Y = np.zeros((count,dictionary_size))

    # Loop over the training examples:
    count = 0
    for i, sent in enumerate(a):
        ans_partial = np.zeros((1,maxlen_input))
        
        # Loop over the positions of the current target output (the current output sequence):
        l = np.where(sent==3)  #  the position of the symbol EOS
        limit = l[0][0]

        for k in range(1,limit+1):
            # Mapping the target output (the next output word) for one-hot codding:
            y = np.zeros((1, dictionary_size))
            y[0, int(sent[k])] = 1

            # preparing the partial answer to input:
            ans_partial[0,-k:] = sent[0:k]

            # training the model for one epoch using teacher forcing:
            Q[count, :] = q[i:i+1] 
            A[count, :] = ans_partial 
            Y[count, :] = y
            count += 1

    p = model_discrim.predict([ Q, A, Y])
    p = p[-sa:-1]
    P = np.sum(np.log(p))/sa
    
    return P

print('Starting the model...')

# *******************************************************************
# Keras model of the chatbot: 
# *******************************************************************

ad = Adam(lr=learning_rate) 

input_context = Input(shape=(maxlen_input,), dtype='int32', name='the context text')
input_answer = Input(shape=(maxlen_input,), dtype='int32', name='the answer text up to the current token')
LSTM_encoder = LSTM(sentence_embedding_size, kernel_initializer= 'lecun_uniform', name='Encode context')
LSTM_decoder = LSTM(sentence_embedding_size, kernel_initializer= 'lecun_uniform', name='Encode answer up to the current token')

Shared_Embedding = Embedding(output_dim=word_embedding_size, input_dim=dictionary_size, input_length=maxlen_input, name='Shared')
word_embedding_context = Shared_Embedding(input_context)
context_embedding = LSTM_encoder(word_embedding_context)

word_embedding_answer = Shared_Embedding(input_answer)
answer_embedding = LSTM_decoder(word_embedding_answer)

merge_layer = concatenate([context_embedding, answer_embedding], axis=1, name='concatenate the embeddings of the context and the answer up to current token')
out = Dense(dictionary_size/2, activation="relu", name='relu activation')(merge_layer)
out = Dense(dictionary_size, activation="softmax", name='likelihood of the current token using softmax activation')(out)

model = Model(inputs=[input_context, input_answer], outputs = [out])

model.compile(loss='categorical_crossentropy', optimizer=ad)

# Loading the data:
vocabulary = cPickle.load(open(vocabulary_file, 'rb'))

print("\n \n \n \n    CHAT:     \n \n")

# Processing the user query:
prob = 0
que = ''
last_query  = ' '
last_last_query = ''
text = ' '
last_text = ''
print('computer: hi ! please type your name.\n')
name = raw_input('user: ')
print('computer: hi , ' + name +' ! My name is ' + name_of_computer + '.\n') 


while que <> 'exit .':
    
    que = raw_input('user: ')
    que = preprocess(que, name_of_computer)
    # Collecting data for training:
    q = last_query + ' ' + text
    a = que
    qf.write(q + '\n')
    af.write(a + '\n')
    # Composing the context:
    if prob > 0.2:
        query = text + ' ' + que
    else:    
        query = que
   
    last_text = text
    
    Q = tokenize(query)
    
    # Using the trained model to predict the answer:
    model.load_weights(weights_file)
    predout, prob = greedy_decoder(Q[0:1])
    start_index = predout.find('EOS')
    text = preprocess(predout[0:start_index], name) + ' EOS'
    
    model.load_weights(weights_file_GAN)
    predout, prob2 = greedy_decoder(Q[0:1])
    start_index = predout.find('EOS')
    text2 = preprocess(predout[0:start_index], name) + ' EOS'
    
    p1 = run_discriminator(Q, tokenize(text))
    p2 = run_discriminator(Q, tokenize(text2))
    
    if max([prob, prob2]) > .9:
        if prob > prob2:
            best = text[0 : -4]
        else:
            best = text2[0 : -4]
    else:
        if p1 > p2:
            best = text[0 : -4]
        else:
            best = text2[0 : -4]
    init = ''

    print('\n' + 'computer: ' + best)
    
    last_last_query = last_query    
    last_query = que

qf.close()
af.close()
