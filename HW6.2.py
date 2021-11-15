#!/usr/bin/env python
# coding: utf-8

# In[10]:


from keras.datasets import mnist
import numpy as np
import keras
from keras import layers
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.callbacks import EarlyStopping
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Activation


# In[7]:


(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))
print(x_train.shape)
print(x_test.shape)


# In[8]:


(fashion_x_train, fashion_y_train), (fashion_x_test, fashion_y_test) = tf.keras.datasets.fashion_mnist.load_data()
fashion_x_train = fashion_x_train.astype('float32') / 255.
fashion_x_test = fashion_x_test.astype('float32') / 255.
fashion_x_train = np.reshape(fashion_x_train, (len(fashion_x_train), 28, 28, 1))
fashion_x_test = np.reshape(fashion_x_test, (len(fashion_x_test), 28, 28, 1))


# In[11]:


input_img = keras.Input(shape=(28, 28, 1))

x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

# at this point the representation is (4, 4, 8) i.e. 128-dimensional

x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(16, (3, 3), activation='relu')(x)
x = layers.UpSampling2D((2, 2))(x)
decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = keras.Model(input_img, decoded)
autoencoder.compile(optimizer='rmsprop',
                loss='mean_squared_error')

es_cb = EarlyStopping(monitor='val_loss', patience=2, verbose=1, mode='auto')
history = autoencoder.fit(x_train, x_train,
                epochs=50,
                batch_size=128,
                shuffle=True,
                callbacks=es_cb,
                validation_split=0.2)


# In[12]:


plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.legend()
plt.show()


# In[13]:


train_x_reconstructions = autoencoder.predict(x_train)
mse = np.mean(np.power(x_train - train_x_reconstructions, 1), axis=1)


# In[14]:


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


# In[15]:


threshold = 4*autoencoder.evaluate(x_train,x_train,batch_size=x_train.shape[0])


# In[28]:


# mse = np.mean(np.power(x_train - train_x_reconstructions, 1), axis=1)


# In[40]:


anomaly = 0
for i in range(len(mse)):
    if abs(np.mean(mse[i])) > threshold:
        anomaly += 1
print(f"Detected {anomaly:,} outliers in a total of {len(mse):,} transactions [{anomaly/len(mse):.2%}].")


# In[34]:


(fashion_x_train, fashion_y_train), (fashion_x_test, fashion_y_test) = tf.keras.datasets.fashion_mnist.load_data()
fashion_x_train = fashion_x_train.astype('float32') / 255.
fashion_x_test = fashion_x_test.astype('float32') / 255.
fashion_x_train = np.reshape(fashion_x_train, (len(fashion_x_train), 28, 28, 1))
fashion_x_test = np.reshape(fashion_x_test, (len(fashion_x_test), 28, 28, 1))
 


# In[35]:


fashion_train_reconstructions = autoencoder.predict(fashion_x_train)
# threshold_fashion = 4*autoencoder.evaluate(fashion_x_train,fashion_x_train,batch_size=fashion_x_train.shape[0])
mse_fashion = np.mean(np.power(fashion_x_train - fashion_train_reconstructions, 1), axis=1)


# In[39]:


anomaly_fashion = 0
for i in range(len(mse)):
    if abs(np.mean(mse_fashion[i])) > threshold:
        anomaly_fashion += 1
print(f"Detected {anomaly_fashion:,} outliers in a total of {len(mse_fashion):,} transactions [{anomaly_fashion/len(mse_fashion):.2%}].")


# In[37]:


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


# In[38]:


# saving whole model
autoencoder.save('HW6.2-autoencoder')


# In[ ]:




