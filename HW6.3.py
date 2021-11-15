#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.datasets import mnist
import numpy as np
import keras
from keras import layers
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.callbacks import EarlyStopping
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Activation


# In[13]:



(x_train, y_train), (x_test, y_test) =  tf.keras.datasets.cifar10.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 32, 32, 3))
x_test = np.reshape(x_test, (len(x_test), 32, 32, 3))
print(x_train.shape)
print(x_test.shape)


# In[14]:



x_train = x_train[0:30000]


# In[15]:


print(x_train.shape)


# In[16]:


input_img = keras.Input(shape=(32, 32, 3))

x = Conv2D(64, (3, 3), padding='same')(input_img)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(32, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(16, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

x = Conv2D(16, (3, 3), padding='same')(encoded)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(64, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(3, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
decoded = Activation('sigmoid')(x)

autoencoder = keras.Model(input_img, decoded)
autoencoder.compile(optimizer='rmsprop',
                loss='mean_squared_error')

es_cb = EarlyStopping(monitor='val_loss', patience=2, verbose=1, mode='auto')
history = autoencoder.fit(x_train, x_train,
                epochs=50,
                batch_size=512,
                callbacks=es_cb,
                shuffle=True,
                validation_split=0.2)


# In[23]:


# https://github.com/shibuiwilliam/Keras_Autoencoder/blob/master/Cifar_Conv_AutoEncoder.ipynb
# saving whole model
autoencoder.save('HW6.3-autoencoder')


# In[18]:


plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.legend()
plt.show()


# In[20]:


n = 10
plt.figure(figsize=(20, 4))
for i in range(1, n + 1):
    # Display original
    ax = plt.subplot(2, n, i)
    plt.imshow(x_train[i].reshape(32, 32,3))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display reconstruction
    ax = plt.subplot(2, n, i + n)
    plt.imshow(train_x_reconstructions[i].reshape(32, 32,3))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()


# In[19]:


train_x_reconstructions = autoencoder.predict(x_train)
mse = np.mean(np.power(x_train - train_x_reconstructions, 1), axis=1)


# In[21]:


threshold = 4*autoencoder.evaluate(x_train,x_train,batch_size=x_train.shape[0])


# In[44]:


anomaly = 0
for i in range(len(mse)):
    if abs(np.mean(mse[i])) > threshold:
        anomaly += 1
print(f"Detected {anomaly:,} outliers in a total of {len(mse):,} transactions [{anomaly/len(mse):.2%}].")


# In[31]:


from keras.datasets import cifar100

(x_train100, y_train100), (x_test100, y_test100) = cifar100.load_data()
x_train100 = x_train100.astype('float32') / 255.
x_test100 = x_test100.astype('float32') / 255.
x_train100 = np.reshape(x_train100, (len(x_train100), 32, 32, 3))
x_test100 = np.reshape(x_test100, (len(x_test100), 32, 32, 3))
print(x_train100.shape)
print(x_test100.shape)


# In[32]:


#remove examples with y=58(truck)
x_train100, y_train100 = zip(*((x, y) for x, y in zip(x_train100, y_train100) if y != 58))


# In[33]:


x_train100 = np.asarray(x_train100) 
print(x_train100.shape)


# In[34]:


x_train100 = x_train100[0:30000]


# In[35]:


cifar100_train_reconstructions = autoencoder.predict(x_train100)
# threshold_fashion = 4*autoencoder.evaluate(fashion_x_train,fashion_x_train,batch_size=fashion_x_train.shape[0])
mse_cifar100 = np.mean(np.power(x_train100 - cifar100_train_reconstructions, 1), axis=1)


# In[43]:


anomaly_100 = 0
for i in range(len(mse)):
    if abs(np.mean(mse_cifar100[i])) > threshold:
        anomaly_100 += 1
print(f"Detected {anomaly_100:,} outliers in a total of {len(mse_cifar100):,} transactions [{anomaly_100/len(mse_cifar100):.2%}].")


# In[42]:


np.mean(mse_cifar100[1])


# In[41]:


n = 10
plt.figure(figsize=(20, 4))
for i in range(1, n + 1):
    # Display original
    ax = plt.subplot(2, n, i)
    plt.imshow(x_train100[i].reshape(32, 32,3))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display reconstruction
    ax = plt.subplot(2, n, i + n)
    plt.imshow(cifar100_train_reconstructions[i].reshape(32, 32,3))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()


# In[ ]:




