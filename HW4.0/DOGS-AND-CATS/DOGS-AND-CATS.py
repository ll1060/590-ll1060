#!/usr/bin/env python
# coding: utf-8

# In[14]:


import os, shutil
from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator

import warnings
import random
# from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.utils import to_categorical


# In[8]:


# copy from original directory to a new directory
# and split train data in original directory into train, validation and test
# leave original test folder as it is
original_dataset_dir = '/Users/lingfengcao/Leyao/ANLY590/dogs-vs-cats/train' #original path for data
base_dir = '/Users/lingfengcao/Leyao/ANLY590/dogs-vs-cats-train-test/' 
train_dir = os.path.join(base_dir, 'train')
# print(train_dir)
validation_dir = os.path.join(base_dir, 'validation')
# os.makedirs(validation_dir)
test_dir = os.path.join(base_dir, 'test')
# os.makedirs(test_dir)
train_cats_dir = os.path.join(train_dir, 'cats')
# os.makedirs(train_cats_dir)
train_dogs_dir = os.path.join(train_dir, 'dogs')
# os.makedirs(train_dogs_dir)
validation_cats_dir = os.path.join(validation_dir, 'cats')
# os.makedirs(validation_cats_dir)
validation_dogs_dir = os.path.join(validation_dir, 'dogs')
# os.makedirs(validation_dogs_dir)
test_cats_dir = os.path.join(test_dir, 'cats')
# os.makedirs(test_cats_dir)
test_dogs_dir = os.path.join(test_dir, 'dogs')
# os.makedirs(test_dogs_dir)
#  copy from original directory to a new directory
#  split cat and dog
#  num train: 1000
# num test: 500
# num val: 500
fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_cats_dir, fname)
    shutil.copyfile(src, dst)
    fnames = ['cat.{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_cats_dir, fname)
    shutil.copyfile(src, dst)
    fnames = ['cat.{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_cats_dir, fname)
    shutil.copyfile(src, dst)
    fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_dogs_dir, fname)
    shutil.copyfile(src, dst)
    fnames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_dogs_dir, fname)
    shutil.copyfile(src, dst)
    fnames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_dogs_dir, fname)
    shutil.copyfile(src, dst)


# In[15]:


# intialize model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',
input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# configure model 
model.compile(loss='binary_crossentropy',
optimizer=optimizers.RMSprop(lr=1e-4),
metrics=['acc'])


# In[17]:


# transform raw image data to arrays
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
                                                        train_dir,
                                                        target_size=(150, 150),
                                                        batch_size=20,
                                                        class_mode='binary')
validation_generator = test_datagen.flow_from_directory(
                                                        validation_dir, 
                                                        target_size=(150, 150),
                                                        batch_size=20,
                                                        class_mode='binary')


# In[18]:


# fit model and save training and validation metrics
history = model.fit_generator(
train_generator,
steps_per_epoch=100,
epochs=30,
validation_data=validation_generator,
validation_steps=50)


# In[20]:


# plot the saved metrics to visualize train and validation loss
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


# In[23]:


img_path = '/Users/lingfengcao/Leyao/ANLY590/dogs-vs-cats-train-test/test/cats/cat.1700.jpg'
from keras.preprocessing import image
img = image.load_img(img_path, target_size=(150, 150))
img_tensor = image.img_to_array(img) #convert from raw image to np array
img_tensor = np.expand_dims(img_tensor, axis=0) #convert in to 4d tensor
img_tensor /= 255.
print(img_tensor.shape)
plt.imshow(img_tensor[0])
plt.show()


# In[27]:


from keras import models
layer_outputs = [layer.output for layer in model.layers[:8]] #Extracts the outputs of the first eight layers
activation_model = models.Model(inputs=model.input, outputs=layer_outputs) # given input, returns model output
activations = activation_model.predict(img_tensor) #return a list og np arrays of layer activation
layer_names = []
for layer in model.layers[1:7]:
    layer_names.append(layer.name) # Names of the layers, so you can have them as part of your plot

images_per_row = 16
for layer_name, layer_activation in zip(layer_names, activations): # Displays the feature maps
    n_features = layer_activation.shape[-1] # Number of features in the feature map
    size = layer_activation.shape[1] #The feature map has shape (1, size, size, n_features).
    n_cols = n_features // images_per_row # Tiles the activation channels in this matrix
    display_grid = np.zeros((size * n_cols, images_per_row * size))
    for col in range(n_cols): # Tiles each filter into a big horizontal grid
        for row in range(images_per_row):
            channel_image = layer_activation[0, :, :, col * images_per_row + row]
            channel_image -= channel_image.mean() # Post-processes the feature to make it visually palatable
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[col * size : (col + 1) * size, # Displays the grid
                         row * size : (row + 1) * size] = channel_image
    scale = 1. / size
    plt.figure(figsize=(scale * display_grid.shape[1],
                        scale * display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')


# In[ ]:




