#!/usr/bin/env python
# coding: utf-8

# In[2]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from keras.datasets import imdb


# In[16]:



(train_data, train_targets), (test_val_data, test_val_targets) = imdb.load_data(
num_words=10000)


# In[17]:


# split test and validation to be 50:50
test_data, val_data, test_targets, val_targets = train_test_split(test_val_data, test_val_targets, test_size=0.5)


# In[24]:



def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)
x_val = vectorize_sequences(val_data)


# In[25]:


y_train = np.asarray(train_targets).astype('float32')
y_test = np.asarray(test_targets).astype('float32')
y_val =  np.asarray(val_targets).astype('float32')


# In[56]:


from keras import models
from keras import layers
model = models.Sequential()
model.add(layers.Dense(32, activation='relu', input_shape=(10000,)))
model.add(layers.Dropout(0.2, noise_shape=None, seed=None))
model.add(layers.Dense(32, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)))
model.add(layers.Dense(50, activation = "relu"))
model.add(layers.Dense(1, activation='sigmoid'))


# In[57]:


from keras import optimizers
from keras import losses
from keras import metrics
model.compile(optimizer = "adam",
loss=losses.binary_crossentropy,
metrics=['acc'])


# In[58]:


history = model.fit(x_train,
y_train,
epochs=20,
batch_size=512,
validation_data=(x_val, y_val))


# In[59]:


history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, 21)
plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[60]:


# get predictions 
scores = model.evaluate(x_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))


# In[ ]:




