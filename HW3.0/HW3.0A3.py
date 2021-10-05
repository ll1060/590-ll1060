#!/usr/bin/env python
# coding: utf-8

# In[12]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from keras import models
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from keras.datasets import imdb


# In[5]:


from keras.datasets import reuters
(train_data, train_targets), (test_val_data, test_val_targets) = reuters.load_data(
num_words=10000)


# In[6]:


# split test and validation to be 50:50
test_data, val_data, test_targets, val_targets = train_test_split(test_val_data, test_val_targets, test_size=0.5)


# In[7]:


word_index = reuters.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
decoded_newswire = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])


# In[9]:


def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)
x_val = vectorize_sequences(val_data)


# In[10]:


def to_one_hot(labels, dimension=46):
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, label] = 1.
    return results
one_hot_train_labels = to_one_hot(train_targets)
one_hot_test_labels = to_one_hot(test_targets)
one_hot_val_labels = to_one_hot(val_targets)


# In[18]:



model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,), kernel_regularizer=keras.regularizers.l2(0.001)))
model.add(layers.Dense(64, activation='relu', kernel_regularizer='l1_l2'))
model.add(layers.Dense(46, activation='softmax'))

model.compile(optimizer='rmsprop',
loss='categorical_crossentropy',
metrics=['accuracy'])


# In[19]:


history = model.fit(x_train,one_hot_train_labels,
epochs=20,
batch_size=512,
validation_data=(x_val, one_hot_val_labels))


# In[20]:


import matplotlib.pyplot as plt
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[21]:


results = model.evaluate(x_test, one_hot_test_labels)
results


# In[ ]:




