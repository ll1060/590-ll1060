#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

print(tf.__version__)


# In[2]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split


# In[3]:


# load data
from keras.datasets import boston_housing

# train and (test&val) split
(train_data, train_targets), (test_val_data, test_val_targets) = tf.keras.datasets.boston_housing.load_data(test_split=0.2)


# In[4]:


# split test and validation to be 50:50
test_data, val_data, test_targets, val_targets = train_test_split(test_val_data, test_val_targets, test_size=0.5)


# In[5]:


# normalize the data

# normalize x in train, test and val
mean_x = train_data.mean(axis=0)
train_data -= mean_x
std_x = train_data.std(axis=0)
train_data /= std_x
test_data -= mean_x
test_data /= std_x
val_data -= mean_x
val_data /= std_x

# normalize y in train, test and val
mean_y = train_targets.mean(axis=0)
train_targets -= mean_y
std_y = train_targets.std(axis=0)
train_targets /= std_y
test_targets -= mean_y
test_targets /= std_y
val_targets -= mean_y
val_targets /= std_y


# In[6]:


# initialize ANN model
from keras import models
from keras import layers
from tensorflow.keras.regularizers import l2

def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu',
    input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model


# In[8]:


import numpy as np
k=4
num_val_samples = len(train_data) // k
num_epochs = 300
all_val_loss_histories = []
all_train_loss_histories = []
for i in range(k):
    print('processing fold #', i)
    # get validation data split from the training set
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
    partial_train_data = np.concatenate(
    [train_data[:i * num_val_samples],
    train_data[(i + 1) * num_val_samples:]],
    axis=0)
    partial_train_targets = np.concatenate(
    [train_targets[:i * num_val_samples],
    train_targets[(i + 1) * num_val_samples:]],
    axis=0)
    
    # initialize model
    model = build_model()
    
    # 
    history = model.fit(partial_train_data, partial_train_targets,
    validation_data=(val_data, val_targets),
    epochs=num_epochs, batch_size=1, verbose=0)
    val_loss_history = history.history['val_loss']
    all_val_loss_histories.append(val_loss_history)
    train_loss_history = history.history['loss']
    all_train_loss_histories.append(train_loss_history)


# In[9]:



# visualize train and validation loss during each fold of k-fold cv
for i in range(0,4):
    epochs = range(1, len(all_train_loss_histories[i]) + 1)
    plt.plot(epochs, all_train_loss_histories[i], "bo", label="Training loss")
    plt.plot(epochs, all_val_loss_histories[i], "r", label="Validation loss")
    plt.title(str(i+1) + " Fold Training and validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


# In[10]:


# get predictions 
yp=model.predict(test_data)
yp_val=model.predict(val_data) 


# In[11]:


#UN-NORMALIZE DATA (CONVERT BACK TO ORIGINAL UNITS)
train_data = std_x * train_data + mean_x 
val_data = std_x * val_data + mean_x 
train_targets = std_y * train_targets + mean_y 
val_targets = std_y * val_targets + mean_y 
yp = std_y * yp + mean_y 
yp_val=std_y * yp_val + mean_y 


# In[17]:


test_targets = std_y*test_targets + mean_y


# In[18]:


#PARITY PLOT
FS = 18
plt.plot(yp,yp,'-')
plt.plot(test_targets,yp,'o')
plt.xlabel("y (predicted)", fontsize=FS)
plt.ylabel("y (data)", fontsize=FS)
plt.show()
plt.clf()


# In[ ]:




