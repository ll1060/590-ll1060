#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.datasets import mnist
import numpy as np
import keras
from keras import layers
import matplotlib.pyplot as plt
import tensorflow as tf


# In[2]:


(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print(x_train.shape)
print(x_test.shape)


# In[3]:


input_img = keras.Input(shape=(784,))
encoded = layers.Dense(128, activation='relu')(input_img)
encoded = layers.Dense(64, activation='relu')(encoded)
encoded = layers.Dense(32, activation='relu')(encoded)

decoded = layers.Dense(64, activation='relu')(encoded)
decoded = layers.Dense(128, activation='relu')(decoded)
decoded = layers.Dense(784, activation='sigmoid')(decoded)


# In[4]:


autoencoder = keras.Model(input_img, decoded)
autoencoder.compile(optimizer='rmsprop',
                loss='mean_squared_error')

history = autoencoder.fit(x_train, x_train,
                epochs=100,
                batch_size=256,
                shuffle=True,
                validation_split=0.2)


# In[5]:


plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.legend()
plt.show()


# In[6]:


train_x_reconstructions = autoencoder.predict(x_train)
mse = np.mean(np.power(x_train - train_x_reconstructions, 1), axis=1)


# In[8]:


n = 10
plt.figure(figsize=(20, 4))
for i in range(1, n + 1):
    # Display original
    ax = plt.subplot(2, n, i)
    plt.imshow(x_train[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display reconstruction
    ax = plt.subplot(2, n, i + n)
    plt.imshow(train_x_reconstructions[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()


# In[9]:


threshold = 4*autoencoder.evaluate(x_train,x_train,batch_size=x_train.shape[0])

# mse = np.mean(np.power(x_train - test_x_reconstructions, 2), axis=1)


# In[10]:


# num_dev_threshold = 3

# def mad_score(points):
#     """https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h.htm """
#     m = np.median(points)
#     ad = np.abs(points - m)
#     mad = np.median(ad)
    
#     return 0.6745 * ad / mad

# z_scores = mad_score(mse)
# outliers = z_scores > num_dev_threshold


# In[11]:


anomaly = list(filter(lambda x: x > threshold, mse))
print(f"Detected {len(anomaly):,} outliers in a total of {len(mse):,} transactions [{len(anomaly)/len(mse):.2%}].")


# In[12]:


(fashion_x_train, fashion_y_train), (fashion_x_test, fashion_y_test) = tf.keras.datasets.fashion_mnist.load_data()
fashion_x_train = fashion_x_train.astype('float32') / 255.
fashion_x_test = fashion_x_test.astype('float32') / 255.
fashion_x_train = fashion_x_train.reshape((len(fashion_x_train), np.prod(fashion_x_train.shape[1:])))
fashion_x_test = fashion_x_test.reshape((len(fashion_x_test), np.prod(fashion_x_test.shape[1:])))


# In[13]:


fashion_train_reconstructions = autoencoder.predict(fashion_x_train)
# threshold_fashion = 4*autoencoder.evaluate(fashion_x_train,fashion_x_train,batch_size=fashion_x_train.shape[0])
mse_fashion = np.mean(np.power(fashion_x_train - fashion_train_reconstructions, 1), axis=1)


# In[14]:



# z_scores_fashion = mad_score(mse_fashion)
# outliers_fashion = z_scores_fashion > num_dev_threshold
anomaly_fashion = list(filter(lambda x: x > threshold, mse_fashion))
print(f"Detected {len(anomaly_fashion):,} outliers in a total of {len(mse_fashion):,} transactions [{len(anomaly_fashion)/len(mse_fashion):.2%}].")


# In[15]:


n = 10
plt.figure(figsize=(20, 4))
for i in range(1, n + 1):
    # Display original
    ax = plt.subplot(2, n, i)
    plt.imshow(fashion_x_train[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display reconstruction
    ax = plt.subplot(2, n, i + n)
    plt.imshow(fashion_train_reconstructions[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)


# In[16]:



# saving whole model
autoencoder.save('HW6.1-autoencoder')


# In[ ]:




