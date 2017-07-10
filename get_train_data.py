# -*- coding: utf-8 -*-

__author__ = 'Oswaldo Ludwig'
__version__ = '1.01'

import numpy as np
np.random.seed(1234)  # for reproducibility
import pandas as pd
import os
import csv
import nltk
import itertools
import operator
import pickle
import numpy as np    
from keras.preprocessing import sequence
from scipy import sparse, io
from numpy.random import permutation
import re
    
questions_file = 'context'
answers_file = 'answers'
vocabulary_file = 'vocabulary_movie'
padded_questions_file = 'Padded_context'
padded_answers_file = 'Padded_answers'
unknown_token = 'something'

vocabulary_size = 7000
max_features = vocabulary_size
maxlen_input = 50
maxlen_output = 50  # cut texts after this number of words

print ("Reading the context data...")
q = open(questions_file, 'r')
questions = q.read()
print ("Reading the answer data...")
a = open(answers_file, 'r')
answers = a.read()
all = answers + questions
print ("Tokenazing the answers...")
paragraphs_a = [p for p in answers.split('\n')]
paragraphs_b = [p for p in all.split('\n')]
paragraphs_a = ['BOS '+p+' EOS' for p in paragraphs_a]
paragraphs_b = ['BOS '+p+' EOS' for p in paragraphs_b]
paragraphs_b = ' '.join(paragraphs_b)
tokenized_text = paragraphs_b.split()
paragraphs_q = [p for p in questions.split('\n') ]
tokenized_answers = [p.split() for p in paragraphs_a]
tokenized_questions = [p.split() for p in paragraphs_q]

### Counting the word frequencies:
##word_freq = nltk.FreqDist(itertools.chain(tokenized_text))
##print ("Found %d unique words tokens." % len(word_freq.items()))
##
### Getting the most common words and build index_to_word and word_to_index vectors:
##vocab = word_freq.most_common(vocabulary_size-1)
##
### Saving vocabulary:
##with open(vocabulary_file, 'w') as v:
##    pickle.dump(vocab, v)

vocab = pickle.load(open(vocabulary_file, 'rb'))


index_to_word = [x[0] for x in vocab]
index_to_word.append(unknown_token)
word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])

print ("Using vocabulary of size %d." % vocabulary_size)
print ("The least frequent word in our vocabulary is '%s' and appeared %d times." % (vocab[-1][0], vocab[-1][1]))

# Replacing all words not in our vocabulary with the unknown token:
for i, sent in enumerate(tokenized_answers):
    tokenized_answers[i] = [w if w in word_to_index else unknown_token for w in sent]
   
for i, sent in enumerate(tokenized_questions):
    tokenized_questions[i] = [w if w in word_to_index else unknown_token for w in sent]

# Creating the training data:
X = np.asarray([[word_to_index[w] for w in sent] for sent in tokenized_questions])
Y = np.asarray([[word_to_index[w] for w in sent] for sent in tokenized_answers])

Q = sequence.pad_sequences(X, maxlen=maxlen_input)
A = sequence.pad_sequences(Y, maxlen=maxlen_output, padding='post')

with open(padded_questions_file, 'w') as q:
    pickle.dump(Q, q)
    
with open(padded_answers_file, 'w') as a:
    pickle.dump(A, a)