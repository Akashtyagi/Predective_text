#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 14:22:02 2019

@author: Akashtyagi
"""
'''
# =============================================================================
# Sources: 
#     Code from --> https://medium.com/@curiousily/making-a-predictive-keyboard-using-recurrent-neural-networks-tensorflow-for-hackers-part-v-3f238d824218
#     Detailed Explanation of steps - https://machinelearningmastery.com/5-step-life-cycle-long-short-term-memory-models-keras/
# ============================================================================='''
import numpy as np
np.random.seed(42)
from keras.models import Sequential,load_model
from keras.layers import LSTM,Dropout
from keras.layers import Dense
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping
import pickle
import matplotlib.pyplot as plt
import heapq
import time

def play_sound():
    import os
    duration = 1  # seconds
    freq = 330  # Hz
    os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq))

'''
# =============================================================================
# Import Dataset
# ============================================================================='''
path = 'beyond_good_and_evil.txt'
text = open(path).read().lower()
print(len(text))

'''
# =============================================================================
# Extracting parameters of Data
# ============================================================================='''
chars = sorted(list(set(text)))
print("Unique Characters: ",len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

SEQUENCE_LENGTH = 40
step = 3
sentences = []
next_chars = []

for i in range(0, len(text) - SEQUENCE_LENGTH, step):
    sentences.append(text[i: i + SEQUENCE_LENGTH])
    next_chars.append(text[i + SEQUENCE_LENGTH])
print("No of training examples: ",len(sentences))

'''
# =============================================================================
# Preparing input to Neural Network - Word Vectors
# ============================================================================='''
# Creating X,y bool array of dimension equal to 
# X = no. of sentences,length of each sentence, total no. of chars
# y = no. of sentences, unique chars

X = np.zeros((len(sentences), SEQUENCE_LENGTH, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)

# For each sentence we store :
# X = The integer-equivalent-char is set to 1 for each char of sentence.
#      Ex- "This " so for "T" we find what integer is assigned to it and 
#           set that integer at that sentence index = 1
#
# y = The integer-equivalent-char that should appear after the sentence.

for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
#        print(f"( {i},{t},{char_indices[char]} )",end="")
        X[i, t, char_indices[char]] = 1 # For each sentence we have 57 char
#    print("-------------->> ")
    y[i, char_indices[next_chars[i]]] = 1
print(f"X-Shape: {X.shape}   Y-Shape: {y.shape}")

'''
# =============================================================================
# Defining a Model
# ============================================================================='''
model = Sequential()
model.add(LSTM(128,input_shape=(SEQUENCE_LENGTH,len(chars)))) # input-shape= (40,57)
model.add(Dropout(0.2))
model.add(Dense(len(chars),activation='softmax')) # Dense= 57
#model.add(Activation('softmax'))

'''
# =============================================================================
# Compiling a model
# ============================================================================='''
optimizer = RMSprop(lr=0.01) # Using Root-mean-Squared as optimizer for Network with Learing Rate= 0.01
model.compile(loss='categorical_crossentropy', 
              optimizer=optimizer, 
              metrics=['accuracy'])

es = EarlyStopping(monitor="val_acc",mode="max",verbose=1,patience=4)

'''
# =============================================================================
# Fitting the model
# ============================================================================='''
# Batch-size - We cannot input entire dataset to the model in one go as it has around 1Lakh entries,
#            so we break out dataset into batches. So batches are fed to the network and N no. of batches
#            means 1 complete dataset input.
#
# Epoch - No. of times entire dataset get fed to the network 

start = time.time()
history = model.fit(X,y,batch_size=80,epochs=100,validation_split=0.05,shuffle=True,callbacks=[es]).history
print("Time: ",time.time()-start)
play_sound()

# If epoch=20 so 20 times dataset is fed to the model.

'''
# =============================================================================
# Saving the model
# ============================================================================='''
model.save('beyond_good_and_evil_128layer_80batchsize_100epoch_2dropout_01lr_05vs_keras_model.h5')
pickle.dump(history, open("history.p", "wb"))

model.summary()

# Re-loading model
model = load_model('beyond_good_and_evil_128layer_60epoch_2dropout_keras_model.h5')
history = pickle.load(open("history.p", "rb"))

'''
# =============================================================================
# Plotting Graphs
# ============================================================================='''
#   -- Model Accuracy --
plt.plot(history['acc'])
plt.plot(history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left');
plt.savefig('accuracy.png')

#  -- Model Loss --
plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left');
plt.savefig('loss.png')

loss,accuracy = model.evaluate(X,y)

'''
# =============================================================================
# Model Testing
# ============================================================================='''
def prepare_input(text):
    ''' Convert our sentence to input form. '''
    xx = np.zeros((1, SEQUENCE_LENGTH, len(chars)))
    for t, char in enumerate(text):
        xx[0, t, char_indices[char]] = 1.
    return xx

def sample(preds, top_n=3):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds)
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    return heapq.nlargest(top_n, range(len(preds)), preds.take)

def predict_completion(text):
    original_text = text
    completion = ''
    while True:
        x = prepare_input(text)
        preds = model.predict(x, verbose=0)[0]
        next_index = sample(preds, top_n=1)[0]
        next_char = indices_char[next_index]
        text = text[1:] + next_char
        completion += next_char
        if len(original_text + completion) + 2 > len(original_text) and next_char == ' ':
            return completion
        
def predict_completions(text, n=3):
    x = prepare_input(text)
    preds = model.predict(x, verbose=0)[0]  #Verbose= Information displayed on terminal 0=All,1=None,2=Just loss
    next_indices = sample(preds, n)
    return [indices_char[idx] + predict_completion(text[1:] + indices_char[idx]) for idx in next_indices]

quotes = [
    "Those who work honestly have the courage to stand for his rights.",
    "The pain is temporary but the results are permanent.",
    "I'm not upset that you lied to me but the fact that it was done for money hurts me.",
    "And those who were seen dancing were there to celebrate the victory.",
    "It is hard enough to remember my opinions, without also remembering my reasons for them!"
]



for q in quotes:
#    q = "The lady sitting beside me is working there get her work done."
    seq = q[:40].lower()
    print(seq)
    print(predict_completions(seq, 5))
    print()
    
    
    
# =============================================================================
# Results
# =============================================================================
    
'''
**** LSTM=128, DroupOut=0.2,lr=0.01,batch_size=80,epoch=Earlystopping,validation=0.05
  Accuracy= 56
  Loss = 1.6
  Val_acc= 52
those who work honestly have the courage
[' the ', ', ', '--they ', 'd ', 'ness ']

the pain is temporary but the results ar
['e ', 'tists ', 'ound ', 'a ', 'id ']

i'm not upset that you lied to me but th
['e ', 'at ', 'is ', 'ought ', 'rough ']

and those who were seen dancing were the
[' strength ', 're ', 'ir ', 'y ', 'mselves ']

it is hard enough to remember my opinion
[' of ', 's ', ', ', '. ', '--the ']  


**** LSTM=128, DroupOut=0.2,lr=0.01,batch_size=128,epoch=60,validation=0.05
  Accuracy= 63
  Loss = 1.2
  Val_acc= 43
those who work honestly have the courage
[' and ', ', ', 't ', 'r ', 'fulness ']

the pain is temporary but the results ar
['e ', 'tists ', 'row ', 'minging ', 'istitation ']

i'm not upset that you lied to me but th
['e ', 'at ', 'oughts, ', 'ink, ', 'ratherimate ']

and those who were seen dancing were the
[' spirit ', 're ', 'ir ', 'n ', 'y ']

it is hard enough to remember my opinion
[' of ', 's ', ', ', 'ality ', '. ']
  

**** LSTM=80, DroupOut=0.2,lr=0.001,batch_size=80,epoch=80,validation=0.33
  Accuracy= 57%
  Loss= 1.47
those who work honestly have the courage
[' and ', ', ', '. ', 's ', '--and ']

the pain is temporary but the results ar
['e ', 'ing ', 't ', 'reation ', 'd ']

i'm not upset that you lied to me but th
['e ', 'at ', 'is ', 'ough ', 'ut ']

and those who were seen dancing were the
['re ', 'y ', 'm ', 'n ', ' world ']

it is hard enough to remember my opinion
['s ', ' of ', ', ', 'ion ', 'alins ']


**** LSTM=128, DroupOut=0.2,lr=0.01,batch_size=80,epoch=60,validation=0.05
  Accuracy= 63%
  Loss= 1.21
  Val_acc= 46.94
those who work honestly have the courage
[' and ', ', ', 't ', 'r ', 'fulness ']

the pain is temporary but the results ar
['e ', 'tists ', 'row ', 'minging ', 'istitation ']

i'm not upset that you lied to me but th
['e ', 'at ', 'oughts, ', 'ink, ', 'ratherimate ']

and those who were seen dancing were the
[' spirit ', 're ', 'ir ', 'n ', 'y ']

it is hard enough to remember my opinion
[' of ', 's ', ', ', 'ality ', '. ']

*** LSTM=128, DroupOut=0.3,lr=0.01,batch_size=160,epoch=20,validation=0.33

'''