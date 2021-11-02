#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tensorflow import keras
from keras.optimizers import RMSprop
from keras.optimizers import Adam
import tensorflow as tf
import numpy as np
conv_model = keras.models.load_model('./saved_models/1DConv',compile=True)
lstm_model = keras.models.load_model('./saved_models/lstm',compile=True)


# In[2]:


X_train =  np.load('./train_test_data/X_train.npy')
X_test =  np.load('./train_test_data/X_test.npy')
X_train_3d =  np.load('./train_test_data/X_train_3d.npy')
X_test_3d =  np.load('./train_test_data/X_test_3d.npy')
y_train =  np.load('./train_test_data/y_train.npy')
y_test = np.load('./train_test_data/y_test.npy')


# In[3]:


METRICS = [
      keras.metrics.BinaryAccuracy(name='accuracy',),
      keras.metrics.AUC(name='auc',from_logits=True)]

conv_model.compile(optimizer=RMSprop(lr=1e-4),
loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
 metrics=METRICS)


# In[4]:


print('1d Convolutional model training metrics: ', conv_model.evaluate(X_train, y_train))
print('1d Convolutional model testing metrics: ', conv_model.evaluate(X_test,y_test))


# In[5]:


lstm_model.compile(optimizer=RMSprop(lr=1e-4),
loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
 metrics=METRICS)


# In[7]:


print('LSTM model training metrics: ', lstm_model.evaluate(X_train_3d, y_train))
print('LSTM model testing metrics: ', lstm_model.evaluate(X_test_3d,y_test))


# In[ ]:




