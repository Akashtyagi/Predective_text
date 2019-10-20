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
from keras.layers import Dense, Activation
from keras.optimizers import RMSprop
import pickle
import matplotlib.pyplot as plt
import heapq

'''
# =============================================================================
# Import Dataset
# ============================================================================='''
path = 'beyond_good_and_evil.txt'
text = open(path).read().lower()
text = text.replace("\n",' ')
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
# Preparing input to Neural Network
# ============================================================================='''
X = np.zeros((len(sentences), SEQUENCE_LENGTH, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)

for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
#        print(f"( {i},{t},{char_indices[char]} )",end="")
        X[i, t, char_indices[char]] = 1
#    print("-------------->> ")
    y[i, char_indices[next_chars[i]]] = 1
print(f"X-Shape: {X.shape}   Y-Shape: {y.shape}")

'''
# =============================================================================
# Defining a Model
# ============================================================================='''
layers = [LSTM(128,input_shape=(SEQUENCE_LENGTH,len(chars))), Dense(len(chars))]
model = Sequential(layers)
model.add(Dropout(0.2))
model.add(Activation('softmax'))

'''
# =============================================================================
# Compiling a model
# ============================================================================='''
optimizer = RMSprop(lr=0.01) # Using Root-mean-Squared as optimizer for Network with Learing Rate= 0.01
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

'''
# =============================================================================
# Fitting the model
# ============================================================================='''
# Batch-size - We cannot input entire dataset to the model in one go as it has around 1Lakh entries,
#            so we break out dataset into batches. So batches are fed to the network and N no. of batches
#            means 1 complete dataset input.
# Epoch - No. of times entire dataset get fed to the network 
history = model.fit(X,y,batch_size=128,epochs=60,validation_split=0.05,shuffle=True).history
# Here epoch=20 so 20 times dataset is fed to the model.

'''
# =============================================================================
# Saving the model
# ============================================================================='''
model.save('beyond_good_and_evil_128layer_60epoch_2dropout_keras_model.h5')
pickle.dump(history, open("history.p", "wb"))

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

#  -- Model Loss --
plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left');

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
    generated = text
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
    q = "The artis were painting the art and this"
    seq = q[:40].lower()
    print(seq)
    print(predict_completions(seq, 5))
    print()
    

        
