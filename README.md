# Seq2seq Chatbot for Keras
This repository contains a new generative model of chatbot based on seq2seq modeling. Further details on this model can be found in Section 3 of this paper https://arxiv.org/abs/1711.10122v2. In the case of publication using ideas or pieces of code from this repository, please kindly cite this paper.

The trained model available here used a small dataset composed of ~8K pairs of context (the last two utterances of the dialogue up to the current point) and respective response. The data were collected from dialogues of English courses online. This trained model can be fine-tuned using a closed domain dataset to real-world applications.

The canonical seq2seq model became popular in neural machine translation, a task that has different prior probability distributions for the words belonging to the input and output sequences, since the input and output utterances are written in different languages. The architecture presented here assumes the same prior distributions for input and output words. Therefore, it shares an embedding layer (Glove pre-trained word embedding) between the encoding and decoding processes through the adoption of a new model. To improve the context sensitivity, the thought vector (i.e. the encoder output) encodes the last two utterances of the conversation up to the current point. To avoid forgetting the context during the answer generation, the thought vector is concatenated to a dense vector that encodes the incomplete answer generated up to the current point. The resulting vector is provided to dense layers that predict the current token of the answer. See Section 3.1 of our paper for a better insight into the advantages of our model.

The algorithm iterates by including the predicted token into the incomplete answer and feeding it back to the right-hand side input layer of the model shown below. 

![alt tag](https://github.com/oswaldoludwig/Seq2seq-Chatbot-for-Keras/blob/master/model_graph.png)

The following pseudocode explains the algorithm.

![alt tag](https://github.com/oswaldoludwig/Seq2seq-Chatbot-for-Keras/blob/master/Algorithm.png)

The training of this new model converges in few epochs. Using our dataset of 8K training examples, it was required only 100 epochs to reach categorical cross-entropy loss of 0.0318, at the cost of 139 s/epoch running in a GPU GTX980. The performance of this trained model (provided in this repository) seems as convincing as the performance of a vanilla seq2seq model trained on the ~300K training examples of the Cornell Movie Dialogs Corpus, but requires much less computational effort to train.

**To chat with the pre-trained model:**

1. Download the python file "conversation.py", the vocabulary file "vocabulary_movie", and the net weights "my_model_weights20", which can be found here: https://www.dropbox.com/sh/o0rze9dulwmon8b/AAA6g6QoKM8hBEHGst6W4JGDa?dl=0 ;
2. Run conversation.py.
 
**To train a new model or to fine tune on your own data:**

1. If you want to train from the scratch, delete the file my_model_weights20.h5. To fine tune on your data, keep this file;
2. Download the Glove folder 'glove.6B' and include this folder in the directory of the chatbot (you can find this folder here https://nlp.stanford.edu/projects/glove/). This algorithm applies transfer learning by using a pre-trained word embedding, which is fine tuned during the training;
3. Run split_qa.py to split the content of your training data into two files: 'context' and 'answers' and get_train_data.py to store the padded sentences into the files 'Padded_context' and 'Padded_answers';
4. Run train_bot.py to train the chatbot (it is recommended the use of GPU, to do so type: THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,exception_verbosity=high python train_bot.py);

Name your training data as "data.txt". This file must contain one dialogue utterance per line. If your dataset is big, set the variable num_subsets (in line 29 of train_bot.py) to a larger number.

A nice overview of the current implementations of neural conversational models for different frameworks (along with some results) can be found here: https://github.com/nicolas-ivanov/seq2seq_chatbot_links
