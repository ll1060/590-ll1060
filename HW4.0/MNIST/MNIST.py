#!/usr/bin/env python
# coding: utf-8

# In[24]:


#MODIFIED FROM CHOLLETT P120 
from keras import layers 
from keras import models
import numpy as np
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
import os
from keras.models import Model
warnings.filterwarnings("ignore")


# In[32]:


class streamline_graph_classification:
    
    def __init__(self, data_to_use, data_augmentation=False,epochs=20):
        '''
        Initialize a CNN_model object

        Parameter:

        data_to_use: string, either MNIST, MNIST_FASHION or CIFAR

        '''
        self.data_to_use = data_to_use
        self.epochs = epochs
 
            
    def load_and_clean_data(self):
        '''
        load the dataset of interest: either MNIST, MNIST_FASHION or CIFAR
        
        reshape the data into correct size: num_of_data, pixel_x, pixel_y, num_color_channel
        
        train(include validation) and test split
        
        convert labels from integers to categorical
        
        '''
        
        if self.data_to_use == 'MNIST':
            from keras.datasets import mnist
            (self.train_val_images, self.train_val_labels), (self.test_images, self.test_labels) = mnist.load_data()
            
        elif self.data_to_use == 'MNIST_FASHION':
            from keras.datasets import fashion_mnist
            (self.train_val_images, self.train_val_labels), (self.test_images, self.test_labels) = fashion_mnist.load_data()
            
        elif self.data_to_use == 'CIFAR':
            from keras.datasets import cifar10
            (self.train_val_images, self.train_val_labels), (self.test_images, self.test_labels)= cifar10.load_data()
            
        if self.data_to_use == 'MNIST' or self.data_to_use == 'MNIST_FASHION':
            self.train_val_images = self.train_val_images.reshape((60000, 28, 28, 1))
            self.test_images = self.test_images.reshape((10000, 28, 28, 1))
        else:
            self.train_val_images == self.train_val_images.reshape((50000, 32, 32, 3))
            self.test_images = self.test_images.reshape((10000, 32, 32, 3))

        #CONVERTS A CLASS VECTOR (INTEGERS) TO BINARY CLASS MATRIX.
        self.train_val_labels = to_categorical(self.train_val_labels)
        self.test_labels = to_categorical(self.test_labels)
    
    def train_val_split(self,p=0.8):
        '''
        split training data into training set and validation set, default 80:20
        
        '''

        # split train and validation to be 80:20
        self.train_images, self.val_images, self.train_labels, self.val_labels = train_test_split(self.train_val_images, self.train_val_labels, test_size=p)
        
    def random_example_visualization(self):
        '''
        show a random image from training set
        '''
        
        n = random.randint(0,self.train_images.shape[0])

        image=self.train_images[n]

        from skimage.transform import rescale, resize, downscale_local_mean
        image = resize(image, (10, 10), anti_aliasing=True)
        plt.imshow(image, cmap=plt.cm.gray)
        plt.show()
        
    def build_model(self, model_type = 'CNN'):
            self.model = models.Sequential()

            if model_type =='DFF':
                self.model.add(layers.Dense(512, activation='relu', input_shape=self.train_val_images.shape[1:4]))

            elif model_type == 'CNN':
                self.model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.train_val_images.shape[1:4]))
                self.model.add(layers.MaxPooling2D((2, 2)))

                self.model.add(layers.Conv2D(64, (3, 3), activation='relu')) 
                self.model.add(layers.MaxPooling2D((2, 2)))
                self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))

                self.model.add(layers.Flatten())
                self.model.add(layers.Dense(64, activation='relu'))

            self.model.add(layers.Dense(10, activation='softmax'))
            
            #COMPILATION (i.e. choose optimizer, loss, and metrics to monitor)
            self.model.compile(optimizer='rmsprop',
                                loss='categorical_crossentropy',
                                metrics=['accuracy'])
            self.model.summary()
    
    def train_model(self):
        ## use only a portion of train images when debug
        self.train_images = self.train_images[0:10]
        self.train_labels = self.train_labels[0:10]
        self.model_history = self.model.fit(self.train_images, self.train_labels, epochs=self.epochs, validation_data=(self.val_images, self.val_labels))
        self.train_loss, self.train_acc = self.model.evaluate(self.train_images, self.train_labels)
        self.val_loss, self.val_acc = self.model.evaluate(self.val_images, self.val_labels)
        self.test_loss, self.test_acc = self.model.evaluate(self.test_images, self.test_labels,batch_size=self.test_images.shape[0])
        print('train_acc:', self.train_acc)
        print('test_acc:', self.test_acc)
    
    def visualize_train_val_loss(self):
        self.hist_loss = self.model_history.history['loss']
        self.hist_val_loss =self.model_history.history['val_loss']
        fig, ax = plt.subplots()
        ax.plot(self.hist_loss, 'b', label='Training loss')
        ax.plot(self.hist_val_loss, 'bo', label='Validation loss')
        plt.xlabel('epochs', fontsize=18)
        plt.ylabel('loss', fontsize=18)
        plt.legend()
        plt.show()
    
    def save_model(self):
        '''
        save model to current working directory
        '''
        self.cwd = os.getcwd()
        self.model.save(self.cwd)
        
    def load_model(self, path_to_model):
        '''
        function to load saved model, given path to model
        '''
        from keras.models import load_model
        model = load_model(path_to_model)
        
    def vis_intermediate(self):
        layer_outputs = [layer.output for layer in self.model.layers[1:7]]
        activation_model = Model(inputs=self.model.input,outputs=layer_outputs)
        n = random.randint(0,self.test_images.shape[0])
        img=self.test_images[n].reshape(1,28,28,1)
        
        activations = activation_model.predict(img)
        
#         img_tensor = image.img_to_array(img)
#         img_tensor = np.expand_dims(img_tensor, axis=0)
#         img_tensor /= 255.
#         print(img_tensor.shape)
        layer_names = []
        for layer in self.model.layers[1:7]:
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


# In[33]:


a = streamline_graph_classification(data_to_use = 'MNIST')
a.load_and_clean_data()
a.train_val_split()
a.random_example_visualization()
a.build_model()
a.train_model()
a.visualize_train_val_loss()


# In[34]:


a.vis_intermediate()


# In[ ]:




