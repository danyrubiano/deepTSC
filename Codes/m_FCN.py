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

def FCN(X_train, y_train, X_test, y_test, epochs, batch_size, path, earlystop):

    n_timesteps, n_features, n_outputs = X_train.shape[1], X_train.shape[2], y_train.shape[1]

    inputs_layer = Input(shape=(n_timesteps,n_features))
    
    conv1 = Conv1D(filters=128, kernel_size=8, padding='same')(inputs_layer)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation(activation='relu')(conv1)

    conv2 = Conv1D(filters=256, kernel_size=5, padding='same')(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)

    conv3 = Conv1D(128, kernel_size=3,padding='same')(conv2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)

    gap_layer = GlobalAveragePooling1D()(conv3)

    output_layer = Dense(n_outputs, activation='softmax')(gap_layer)

    model = Model(inputs=input_layer, outputs=output_layer)

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
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), verbose=2, callbacks=[es, mc, reduce_lr])
    
    else:
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), verbose=2, callbacks=[mc, reduce_lr])
    
    return model, history