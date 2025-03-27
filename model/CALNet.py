import pickle
# import numpy as np
# import pandas as pd
import tensorflow as tf
from keras import metrics
from keras.layers import *
from keras.models import Model
from keras.optimizers import Adam
# from tensorflow.python.keras.utils import np_utils

def CBAM(x,filter_num,reduction_ratio):
    # Channel Attention
    avgpool = GlobalAveragePooling1D()(x)
    maxpool = GlobalMaxPooling1D()(x)
    # Shared MLP
    Dense_layer1 = Dense(filter_num//reduction_ratio, activation='relu')
    Dense_layer2 = Dense(filter_num, activation='relu')
    avg_out = Dense_layer2(Dense_layer1(avgpool))
    max_out = Dense_layer2(Dense_layer1(maxpool))

    channel = Add()([avg_out, max_out])
    channel = Activation('sigmoid')(channel)
    channel_out = Multiply()([x, channel])

    # Spatial Attention
    avgpool = tf.reduce_mean(channel_out, axis=1, keepdims=True)
    maxpool = tf.reduce_max(channel_out, axis=1, keepdims=True)
    spatial = Concatenate()([avgpool, maxpool])

    spatial = Conv1D(1, kernel_size=8, padding='SAME')(spatial)
    spatial_out = Activation('sigmoid',)(spatial)

    CBAM_out = tf.multiply(channel_out, spatial_out)
    return CBAM_out


input1 = Input(shape=(4000,1))
seq1 = input1

seq1 = Conv1D(64,8, padding='SAME',activation='relu')(seq1)
seq1 = CBAM(seq1,64,8)
seq1 = MaxPooling1D(4)(seq1)

seq1 = Conv1D(128,8, padding='SAME',activation='relu')(seq1)
seq1 = CBAM(seq1,128,8)
seq1 = MaxPooling1D(4)(seq1)

seq1 = Conv1D(256,8, padding='SAME',activation='relu')(seq1)
seq1 = CBAM(seq1,256,8)
seq1 = MaxPooling1D(4)(seq1)

seq1 = LSTM(128, return_sequences=True)(seq1)
seq1 = Flatten()(seq1)

input2 = Input(shape=(1000,1))
seq2 = input2
seq2 = Conv1D(32,8, padding='SAME',activation='relu')(seq2)
seq2 = CBAM(seq2,32,8)
seq2 = MaxPooling1D(4)(seq2)

seq2 = Conv1D(64,8, padding='SAME',activation='relu')(seq2)
seq2 = CBAM(seq2,64,8)
seq2 = MaxPooling1D(4)(seq2)

seq2 = LSTM(64, return_sequences=True)(seq2)
seq2 = Flatten()(seq2)
seq2 = RepeatVector(2)(seq2)
seq2 = Flatten()(seq2)

add = Add()([seq1,seq2])
seq = Dense(1024, activation='sigmoid')(add)
seq = Dropout(0.5)(seq)

output_tensor = Dense(2, activation='softmax')(seq)


ep = 30  #epoch
# metrics
METRICS = [
    metrics.BinaryAccuracy(name='accuracy'),
    metrics.Precision(name='precision'),
    metrics.Recall(name='recall'),
    metrics.AUC(name='auc'),
]

model = Model(inputs=[input1, input2], outputs=output_tensor)
model.compile(loss='binary_crossentropy',optimizer=Adam(1e-3, amsgrad=True), metrics=METRICS)
history = model.fit([f_trainX,g_trainX],f_trainY, epochs=ep,validation_data = ([f_testX,g_testX],f_testY), workers=4, use_multiprocessing=True,batch_size = 128)


