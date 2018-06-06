import os
import sys
import csv
import wave
import copy
import math

import numpy as np
import pandas as pd

from sklearn.preprocessing import label_binarize
from sklearn.cross_validation import StratifiedKFold, KFold, train_test_split
from sklearn.svm import OneClassSVM, SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from keras.models import Sequential, Model
from keras.layers.core import Dense, Activation
from keras.layers import LSTM, Input, Merge, Dropout
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import SGD, Adam, RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint

sys.path.append("../")
from utilities.utils import *

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
%matplotlib inline

from IPython.display import clear_output

params = Constants()
print(params)

batch_size = 64
nb_feat = 34
nb_class = 4
nb_epoch = 80

optimizer = 'Adadelta'

def build_simple_lstm(nb_feat, nb_class, optimizer='Adadelta'):
    model = Sequential()
    model.add(LSTM(512, return_sequences=True, input_shape=(25, nb_feat)))
    model.add(Activation('tanh'))
    model.add(LSTM(256, return_sequences=False))
    model.add(Activation('tanh'))
    model.add(Dropout(0.3))
    model.add(Dense(512))
    model.add(Activation('tanh'))
    model.add(Dropout(0.3))
    model.add(Dense(512))
    model.add(Dropout(0.3))
    model.add(Activation('tanh'))
    model.add(Dense(nb_class))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model

# optimizer = 
model = build_simple_lstm(nb_feat, nb_class)
model.summary()

# X, y, valid_idxs = get_sample(ids=None, take_all=True)
mX_test, my_test, mX_train, my_train, mX_val, my_val = get_all_samples(gender = 'M')
my_train = to_categorical(my_train, params)
my_test = to_categorical(my_test, params)
my_val = to_categorical(my_val, params)

# idxs_train, idxs_test = train_test_split(range(X.shape[0]), test_size=0.2)

mX_test, _ = pad_sequence_into_array(mX_test, maxlen=25)
mX_train, _ = pad_sequence_into_array(mX_train, maxlen=25)
mX_val, _ = pad_sequence_into_array(mX_val, maxlen=16)

early_stopping = EarlyStopping(monitor='val_loss', patience=20) 
checkpointer = ModelCheckpoint(filepath='MALEONLY.hdf5', verbose=1, save_best_only=True)

    
hist = model.fit(mX_train, my_train, 
                 batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, 
                 validation_data=(mX_test, my_test),  callbacks=[early_stopping, checkpointer])

history = hist
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
