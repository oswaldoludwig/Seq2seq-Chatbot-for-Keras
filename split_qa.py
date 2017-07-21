# -*- coding: utf-8 -*-

__author__ = 'Oswaldo Ludwig'
__version__ = '1.01'

import numpy as np

text = open('dialog_simple', 'r')
q = open('context', 'w')
a = open('answers', 'w')
pre_pre_previous_raw=''
pre_previous_raw=''
previous_raw=''
person = ' '
previous_person=' '

l1 = ['won’t','won\'t','wouldn’t','wouldn\'t','’m', '’re', '’ve', '’ll', '’s','’d', 'n’t', '\'m', '\'re', '\'ve', '\'ll', '\'s', '\'d', 'can\'t', 'n\'t', 'B: ', 'A: ', ',', ';', '.', '?', '!', ':', '. ?', ',   .', '. ,', 'EOS', 'BOS', 'eos', 'bos']
l2 = ['will not','will not','would not','would not',' am', ' are', ' have', ' will', ' is', ' had', ' not', ' am', ' are', ' have', ' will', ' is', ' had', 'can not', ' not', '', '', ' ,', ' ;', ' .', ' ?', ' !', ' :', '? ', '.', ',', '', '', '', '']
l3 = ['-', '_', ' *', ' /', '* ', '/ ', '\"', ' \\"', '\\ ', '--', '...', '. . .']

for i, raw_word in enumerate(text):
    pos = raw_word.find('+++$+++')

    if pos > -1:
        person = raw_word[pos+7:pos+10]
        raw_word = raw_word[pos+8:]
    while pos > -1:
        pos = raw_word.find('+++$+++')
        raw_word = raw_word[pos+2:]
        
    raw_word = raw_word.replace('$+++','')
    previous_person = person

    for j, term in enumerate(l1):
        raw_word = raw_word.replace(term,l2[j])
        
    for term in l3:
        raw_word = raw_word.replace(term,' ')
    
    raw_word = raw_word.lower()

    if i>0:
        q.write(pre_previous_raw[:-1] + ' ' + previous_raw[:-1]+ '\n')  # python will convert \n to os.linese
        a.write(raw_word[:-1]+ '\n')
    
    pre_pre_previous_raw = pre_previous_raw    
    pre_previous_raw = previous_raw
    previous_raw = raw_word

q.close()
a.close()
