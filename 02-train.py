#!/usr/bin/env python
# coding: utf-8

# In[4]:


from sklearn.utils import shuffle
import os
import re
import random
import numpy as np
from keras import preprocessing
from sklearn.model_selection import train_test_split
import keras
from tensorflow.keras import regularizers
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import tensorflow as tf


# In[5]:


X = np.load('./cleaned_data/x_data.npy')
Y = np.load('./cleaned_data/y_data.npy')
X, Y = shuffle(X, Y, random_state = 66) #shuffle these two matrices so that they are not in orders of books
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=77)


# In[65]:


# np.save('./train_test_data/X_train',X_train)
# np.save('./train_test_data/X_test',X_test)
# np.save('./train_test_data/y_train',y_train)
# np.save('./train_test_data/y_test',y_test)


# In[6]:


# X_train.shape


# In[15]:


# #APPLY CONVOLUTION
# import tensorflow as tf

# layer1=tf.keras.layers.Conv1D(
# 	1, 2, activation='relu',
#     kernel_initializer="ones",
#     bias_initializer="zeros",
# 	input_shape=X_train_val.shape[1:]
# 	)
# y_train_val = layer1(X_train_val)



# def report(x,y,l):
# 	print("X:"); print(x); 
# 	print("X SHAPE:",X_train_val.shape)
# 	print("KERNAL SHAPE:",l.get_weights()[0].shape)
# 	print("KERNAL:"); print(l.get_weights()[0])
# 	print("Y SHAPE",y_train_val.shape)
# 	print("Y:"); print(y_train_val)
# report(X_train_val,y_train_val,layer1)

# #GET LAYER INFO
# from keras.models import Sequential 
# model= Sequential()
# model.add(layer1)
# model.fit(X_train_val)
# model.summary()


# In[73]:


from keras.models import Sequential
from keras.layers import Flatten
from keras.optimizers import RMSprop
from keras.optimizers import Adam

max_features = 15000
METRICS = [
      keras.metrics.BinaryAccuracy(name='accuracy',),
      keras.metrics.AUC(name='auc',from_logits=True)]

model = Sequential()
model.add(layers.Embedding(max_features, 16, input_length=10))
model.add(layers.Conv1D(64, 2, activation='relu',  kernel_regularizer=tf.keras.regularizers.l1(0.001),
    bias_regularizer=tf.keras.regularizers.l2(0.001),
    activity_regularizer=tf.keras.regularizers.l1_l2(0.001)))
# model.add(layers.MaxPooling1D(3))
# model.add(layers.Conv1D(32, 2, activation='relu'))
model.add(layers.GlobalMaxPooling1D())
model.add(Flatten())
model.add(layers.Dense(1))

model.compile(optimizer=RMSprop(lr=1e-4),
loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
 metrics=METRICS)

history = model.fit(X_train, y_train,
epochs=30,
batch_size=128,
validation_split=0.2)
model.summary()


# In[76]:


epochs = range(1, len(history.history['loss']) + 1)
plt.figure()
plt.plot(epochs, history.history['loss'], 'bo', label='Training loss')
plt.plot(epochs, history.history['val_loss'], 'b', label='Validation loss')
plt.legend()


# In[77]:


model.evaluate(X_test, y_test)
model.save('./saved_models/1DConv')


# In[51]:


X_train_3d = X_train.reshape(len(X_train), 1, X_train.shape[1])
X_test_3d = X_test.reshape(len(X_test), 1, X_test.shape[1])


# In[66]:


# np.save('./train_test_data/X_train_3d',X_train_3d)
# np.save('./train_test_data/X_test_3d',X_test_3d)


# In[78]:


from keras.layers import Dense, SimpleRNN,LSTM,GRU

model_lstm = Sequential()
#COMMENT/UNCOMMENT TO USE RNN, LSTM,GRU
# model.add(SpatialDropout1D(0.2))
model_lstm.add(LSTM(
# model.add(SimpleRNN(
# model.add(GRU(
64, input_shape=(X_train_3d.shape[1:]),
activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(0.001),
    bias_regularizer=tf.keras.regularizers.l2(0.001),
    activity_regularizer=tf.keras.regularizers.l1_l2(0.001))) 
#NEED TO TAKE THE OUTPUT RNN AND CONVERT TO SCALAR 
model.add(Flatten())
model_lstm.add(Dense(units=1, activation='linear',))
model_lstm.compile(optimizer=RMSprop(lr=1e-4),
loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
 metrics=METRICS)

#TRAIN MODEL
history_lstm = model_lstm.fit(
X_train_3d, y_train, 
epochs=30, 
validation_split=0.2)

#HISTORY PLOT
epochs = range(1, len(history.history['loss']) + 1)
plt.figure()
plt.plot(epochs, history_lstm.history['loss'], 'bo', label='Training loss')
plt.plot(epochs, history_lstm.history['val_loss'], 'b', label='Validation loss')
plt.legend()


# In[79]:


model_lstm.evaluate(X_test_3d,y_test)


# In[80]:


model_lstm.save('./saved_models/lstm')


# In[ ]:




