# Keras
import numpy as np
import os
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, Model
from keras.layers import BatchNormalization, TimeDistributed, Input, Dense, Flatten, LSTM, Conv1D, Conv2D,MaxPooling1D, Dropout, Activation, Reshape, ConvLSTM2D, add
from keras.layers.embeddings import Embedding
from keras.layers.pooling import GlobalAveragePooling1D
from keras.layers.merge import concatenate
from keras.utils.vis_utils import plot_model
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau

"""
CNN-LSTM Multi-headed Architecture

Parameters:
	- X_train, X_test -> Time series values divided in train and test data
	- y_train, y test -> Time series classes divided in train and test data
	- epochs -> Number of epochs for learning
	- batch_size -> Size of the batch for training
	- path -> Where it will be stored
	- earlystop -> 0 if don't use early stopping, or 1 otherwise
Ouputs:
	- model -> Last model trained
	- history -> Learning history
"""

def multi_cnn_lstm(X_train, y_train, X_test, y_test, epochs, batch_size, path, earlystop):

    n_timesteps, n_features, n_outputs = X_train.shape[1], X_train.shape[2], y_train.shape[1]

    # head 1
    inputs_layer1 = Input(shape=(n_timesteps,n_features))
    conv1 = Conv1D(filters=64, kernel_size=3, activation='relu')(inputs_layer1)
    drop1 = Dropout(0.5)(conv1)
    pool1 = MaxPooling1D(pool_size=2)(drop1)
    lstm1 = LSTM(100)(pool1)
    
    # head 2
    inputs_layer2 = Input(shape=(n_timesteps,n_features))
    conv2 = Conv1D(filters=64, kernel_size=5, activation='relu')(inputs_layer2)
    drop2 = Dropout(0.5)(conv2)
    pool2 = MaxPooling1D(pool_size=2)(drop2)
    lstm2 = LSTM(100)(pool2)

    # head 3
    inputs_layer3 = Input(shape=(n_timesteps,n_features))
    conv3 = Conv1D(filters=64, kernel_size=11, activation='relu')(inputs_layer3)
    drop3 = Dropout(0.5)(conv3)
    pool3 = MaxPooling1D(pool_size=2)(drop3)
    lstm3 = LSTM(100)(pool3)

    # merge
    merged = concatenate([lstm1, lstm2, lstm3])
    # interpretation
    dense1 = Dense(100, activation='relu')(merged)
    output_layer = Dense(n_outputs, activation='softmax')(dense1)
    
    model = Model(inputs=[inputs_layer1, inputs_layer2, inputs_layer3], outputs=output_layer)

    # Model compile
    opti = Adam(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=opti, metrics=['accuracy'])
    # rate learning
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, min_lr=0.0001)
    # model checkpoint saving the best model based on val_loss 
    mc = ModelCheckpoint(path+'_best_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
    # simple early stopping
    if earlystop == 1:
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=200)
        # fit network
        history = model.fit([X_train,X_train,X_train], y_train, epochs=epochs, batch_size=batch_size, validation_data=([X_test,X_test,X_test], y_test), verbose=2, callbacks=[es, mc, reduce_lr])
    
    else:
        history = model.fit([X_train,X_train,X_train], y_train, epochs=epochs, batch_size=batch_size, validation_data=([X_test,X_test,X_test], y_test), verbose=2, callbacks=[mc, reduce_lr])
    
    return model, history


