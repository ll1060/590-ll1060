#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import matplotlib.pyplot as plt
import numpy as np
import os
import random
from   scipy.optimize import minimize
from scipy.interpolate import make_interp_spline


# In[2]:


class Data:
    
    def __init__(self, filename, my_data=None):
        
        """ Create an instance of a Data

            Attributes:
            filename: name of the file

        """
        self.my_data = my_data
        self.filename = filename
        
    def read_from_json(self):
        
        """ Read in the dataset that is in a JSON file

        """
        
        with open(self.filename) as json_file:
            self.my_data = json.load(json_file)
            
    def train_test_split(self, train_proportion=0.8):
        
        """ Split data into train/test subsets given the proportion of training data
        
        """
        zipped_data = list(zip(self.my_data['is_adult'],self.my_data['x'],self.my_data['y']))
        random.shuffle(zipped_data)
        shuffled_is_adult, shuffled_x, shuffled_y = zip(*zipped_data)
        train_amount = int(len(self.my_data['is_adult'])*train_proportion)
        self.train_is_adult = shuffled_is_adult[0:train_amount]
        self.train_x = shuffled_x[0:train_amount]
        self.train_y = shuffled_y[0:train_amount]
        self.test_is_adult = shuffled_is_adult[train_amount:]
        self.test_x = shuffled_x[train_amount:]
        self.test_y = shuffled_y[train_amount:]
        
    def visualize_data(self):
        
        """ Create multiple plots to visualize the data
        
        """
        fig, ax = plt.subplots()
        ax.plot(self.my_data['x'],self.my_data['y'],'o')
        ax.set_title('plot of age vs. weight')
        ax.set_xlabel('age')
        ax.set_ylabel('weight')
        plt.show()


        plt.hist(self.my_data['x'],density=False,bins=10)
        plt.title('age distribution')
        plt.xlabel('age(years)')
        plt.ylabel('counts') 
        plt.show()
        
        plt.hist(self.my_data['y'],density=False,bins=10)
        plt.title('weight distribution')
        plt.xlabel('weight(lb)')
        plt.ylabel('counts')
        plt.show()
        

        
    def normalize(self):
        std_train_x = np.std(self.train_x)
        mean_train_x = np.mean(self.train_x)
        std_train_y = np.std(self.train_y)
        mean_train_y = np.mean(self.train_y)
        self.normalized_train_x = [(float(i)-mean_train_x)/std_train_x for i in self.train_x]
        self.normalized_test_x = [(float(i)-mean_train_x)/std_train_x for i in self.test_x]
        self.normalized_train_y = [(float(i)-mean_train_y)/std_train_y for i in self.train_y]
        self.normalized_test_y = [(float(i)-mean_train_y)/std_train_y for i in self.test_y]


# In[3]:


my_weight_data = Data(filename='weight.json')
my_weight_data.read_from_json()
my_weight_data.train_test_split()
my_weight_data.visualize_data()


# In[4]:


my_weight_data.normalize()


# In[5]:


# since we want age < 18 of X as our training data, we apply filter to find the train_x that's less than 18

standard_threshold = (18-np.mean(my_weight_data.train_x))/np.std(my_weight_data.train_x) # we want to find normalized_train_x less than normalized(18), which is to apply standard transformation to 18
zipped_normalized_train_data = list(zip(my_weight_data.normalized_train_x, my_weight_data.normalized_train_y)) # zip training_x and training_y so that they are matched after filter

# apply filter to find training_x that is less than standardized 18
normalized_train_filtered = [t for t in zipped_normalized_train_data if t[0] < standard_threshold]

# unzip the filtered trainig_x and training_y
filterd_train_x, filterd_train_y = zip(*normalized_train_filtered)


# In[18]:


# ----- simple linear regression (use age to predict weight)----- #

def linear_reg(x,m,b):
    return m*x + b


lr_iterations=[]
lr_loss_train=[]
lr_loss_val=[]

lr_iteration=0

def linear_reg_loss(lr_po,x,y):
    global lr_iteration, lr_iterations, lr_loss_train, lr_loss_val
    
    lr_iterations.append(lr_iteration)
    lr_pred = []
    for i in range(len(x)):
        lr_this_pred = linear_reg(x[i],lr_po[0],lr_po[1])
        lr_pred.append(lr_this_pred)
    linear_reg_mse = np.square(np.subtract(y,lr_pred)).mean()
    lr_loss_train.append(linear_reg_mse)
    lr_iteration+=1
    return linear_reg_mse

lr_po=np.random.uniform(0,1.,size=2)

lr_res = minimize(linear_reg_loss, lr_po, args=(filterd_train_x, filterd_train_y), method='Nelder-Mead', tol=1e-15,options={'maxiter': 1000, 'disp': True})
lr_popt=lr_res.x
print("OPTIMAL PARAM:",lr_popt)


unnormalized_linear_reg_test_pred = []
for i in range(len(my_weight_data.normalized_test_x)):
    lr_this_test_pred = linear_reg(my_weight_data.normalized_test_x[i],lr_popt[0],lr_popt[1])
    unnormalized_linear_reg_test_pred.append(lr_this_test_pred*np.std(my_weight_data.train_y) + np.mean(my_weight_data.train_y))

unnormalized_linear_reg_train_pred = []
unnormalized_filtered_train_x = []
for i in range(len(filterd_train_x)):
    lr_this_train_pred = linear_reg(filterd_train_x[i],lr_popt[0],lr_popt[1])
    unnormalized_linear_reg_train_pred.append(lr_this_train_pred*np.std(my_weight_data.train_y) + np.mean(my_weight_data.train_y))
    unnormalized_filtered_train_x.append(filterd_train_x[i]*np.std(my_weight_data.train_x) + np.mean(my_weight_data.train_x))

# train loss plot
plt.plot(lr_iterations,lr_loss_train)
plt.title('training loss of linear regression model')
plt.xlabel('optimizer iterations')
plt.ylabel('training loss')
plt.show()

# model plot
plt.scatter(my_weight_data.test_x,unnormalized_linear_reg_test_pred,color='red',label='test set',marker='*', s=40)
plt.scatter(unnormalized_filtered_train_x,unnormalized_linear_reg_train_pred,color='blue',label='train set',marker='1', s=40)
plt.plot(my_weight_data.test_x, unnormalized_linear_reg_test_pred,c='green',linewidth=3)
plt.legend(('model','test set','training set'))
plt.title('linear regression model (age vs. weight)')
plt.xlabel('age (years)')
plt.ylabel('weight (lb)')
plt.show()

# parity plot
plt.plot(my_weight_data.test_y,unnormalized_linear_reg_test_pred,'r.') # test y vs. unnomarlized predictions
plt.plot(my_weight_data.test_y,my_weight_data.test_y,'k-') # diagonal line
plt.title('Parity Plot of Linear Regression', size=20)
plt.xlabel('true weight', size=14)
plt.ylabel('unnomarlized predictions of weight', size=14)
plt.show()


# In[20]:


# ----- logistic regression (use age to predict weight)----- #

def log_reg(x,A,b,w,s):
    z= (x-b)/w
    return A/(1+np.exp(-z)) + s


iterations=[]
loss_train=[]
loss_val=[]

iteration=0

def log_reg_loss(po,x,y):
    global iteration, iterations, loss_train, loss_val
    
    iterations.append(iteration)
    pred = []
    for i in range(len(x)):
        this_pred = log_reg(x[i],po[0],po[1],po[2],po[3])
        pred.append(this_pred)
    log_reg_mse = np.square(np.subtract(y,pred)).mean()
    loss_train.append(log_reg_mse)
    iteration+=1
    return log_reg_mse

po=np.random.uniform(0,1.,size=4)

res = minimize(log_reg_loss, po, args=(my_weight_data.normalized_train_x, my_weight_data.normalized_train_y), method='Nelder-Mead', tol=1e-15,options={'maxiter': 1000, 'disp': True})
popt=res.x
print("OPTIMAL PARAM:",popt)

# train loss plot
plt.plot(iterations,loss_train)
plt.title('training loss of logistic regression model')
plt.xlabel('optimizer iterations')
plt.ylabel('training loss')
plt.show()

unnormalized_log_reg_test_pred = []
for i in range(len(my_weight_data.normalized_test_x)):
    this_test_pred = log_reg(my_weight_data.normalized_test_x[i],popt[0],popt[1],popt[2],popt[3])
    unnormalized_log_reg_test_pred.append(this_test_pred*np.std(my_weight_data.train_y) + np.mean(my_weight_data.train_y))

unnormalized_log_reg_train_pred = []
for i in range(len(my_weight_data.normalized_train_x)):
    this_train_pred = log_reg(my_weight_data.normalized_train_x[i],popt[0],popt[1],popt[2],popt[3])
    unnormalized_log_reg_train_pred.append(this_train_pred*np.std(my_weight_data.train_y) + np.mean(my_weight_data.train_y))
    
    
array_train_x = np.asarray(my_weight_data.train_x)
X_Y_Spline = make_interp_spline(np.sort(array_train_x), np.sort(unnormalized_log_reg_train_pred))
X_ = np.linspace(min(np.sort(array_train_x)), max(np.sort(array_train_x)),200) 

Y_ = X_Y_Spline(X_)


plt.scatter(my_weight_data.test_x,unnormalized_log_reg_test_pred,color='red',label='test set',marker='*', s=40)
plt.scatter(list(my_weight_data.train_x),unnormalized_log_reg_train_pred,color='blue',label='train set',marker='1', s=40)

plt.plot(X_, Y_,c='green',linewidth=3)
# plt.plot(my_weight_data.test_x,unnormalized_log_reg_test_pred,color='green')
plt.legend(('model','test set','training set'))
plt.title('logistic regression model (age vs. weight)')
plt.xlabel('age (years)')
plt.ylabel('weight (lb)')
plt.show()


# parity plot
plt.plot(my_weight_data.test_y,unnormalized_log_reg_test_pred,'r.') # test y vs. unnomarlized predictions
plt.plot(my_weight_data.test_y,my_weight_data.test_y,'k-') # diagonal line
plt.title('Parity Plot of Logistic Regression (age vs weight)', size=20)
plt.xlabel('true weight', size=14)
plt.ylabel('unnomarlized predictions of weight', size=14)
plt.show()


# In[237]:


# unnormalized_log_reg_test_pred = []
# for i in range(len(my_weight_data.normalized_test_x)):
#     this_test_pred = log_reg(my_weight_data.normalized_test_x[i],popt[0],popt[1],popt[2],popt[3])
#     unnormalized_log_reg_test_pred.append(this_test_pred*np.std(my_weight_data.train_y) + np.mean(my_weight_data.train_y))

# unnormalized_log_reg_train_pred = []
# for i in range(len(my_weight_data.normalized_train_x)):
#     this_train_pred = log_reg(my_weight_data.normalized_train_x[i],popt[0],popt[1],popt[2],popt[3])
#     unnormalized_log_reg_train_pred.append(this_train_pred*np.std(my_weight_data.train_y) + np.mean(my_weight_data.train_y))
    
    


# array_train_x = np.asarray(my_weight_data.train_x)
# X_Y_Spline = make_interp_spline(np.sort(array_train_x), np.sort(unnormalized_log_reg_train_pred))
# X_ = np.linspace(min(np.sort(array_train_x)), max(np.sort(array_train_x)),200) 

# Y_ = X_Y_Spline(X_)


# plt.scatter(my_weight_data.test_x,unnormalized_log_reg_test_pred,color='red',label='test set',marker='*', s=40)
# plt.scatter(list(my_weight_data.train_x),unnormalized_log_reg_train_pred,color='blue',label='train set',marker='1', s=40)

# plt.plot(X_, Y_,c='green',linewidth=3)
# # plt.plot(my_weight_data.test_x,unnormalized_log_reg_test_pred,color='green')
# plt.legend(('model','test set','training set'))
# plt.title('logistic regression model (age vs. weight)')
# plt.xlabel('age (years)')
# plt.ylabel('weight (lb)')
# plt.show()


# # parity plot
# plt.plot(my_weight_data.test_y,unnormalized_log_reg_test_pred,'r.') # test y vs. unnomarlized predictions
# plt.plot(my_weight_data.test_y,my_weight_data.test_y,'k-') # diagonal line
# plt.title('Parity Plot of Logistic Regression (age vs weight)', size=20)
# plt.xlabel('true weight', size=14)
# plt.ylabel('unnomarlized predictions of weight', size=14)
# plt.show()


# In[21]:


# ----- logistic regression (use weight to predict age)----- #
iterations_inv=[]
loss_train_inv=[]
loss_val_inv=[]

iteration_inv=0

def log_reg_loss_inv(po,x,y):
    global iteration_inv, iterations_inv, loss_train_inv, loss_val_inv
    
    iterations_inv.append(iteration_inv)
    pred_inv = []
    for i in range(len(x)):
        this_pred = log_reg(x[i],po[0],po[1],po[2],po[3])
        pred_inv.append(this_pred)
    log_reg_mse_inv = np.square(np.subtract(y,pred_inv)).mean()
    loss_train_inv.append(log_reg_mse_inv)
    iteration_inv+=1
    return log_reg_mse_inv

po_inv=np.random.uniform(0,1.,size=4)

res_inv = minimize(log_reg_loss_inv, po_inv, args=(my_weight_data.normalized_train_y, my_weight_data.normalized_train_x), method='Nelder-Mead', tol=1e-15,options={'maxiter': 1000, 'disp': True})
popt_inv=res_inv.x
print("OPTIMAL PARAM:",popt_inv)



plt.plot(iterations_inv,loss_train_inv)
plt.title('training loss of logistic regression model (weight vs age)')
plt.xlabel('optimizer iterations')
plt.ylabel('training loss')
plt.show()

unnormalized_log_reg_test_pred_inv = []
for i in range(len(my_weight_data.normalized_test_y)):
    this_test_pred_inv = log_reg(my_weight_data.normalized_test_y[i],popt_inv[0],popt_inv[1],popt_inv[2],popt_inv[3])
    unnormalized_log_reg_test_pred_inv.append(this_test_pred_inv*np.std(my_weight_data.train_x) + np.mean(my_weight_data.train_x))

unnormalized_log_reg_train_pred_inv = []
for i in range(len(my_weight_data.normalized_train_y)):
    this_train_pred_inv = log_reg(my_weight_data.normalized_train_y[i],popt_inv[0],popt_inv[1],popt_inv[2],popt_inv[3])
    unnormalized_log_reg_train_pred_inv.append(this_train_pred_inv*np.std(my_weight_data.train_x) + np.mean(my_weight_data.train_x))

    
array_train_x_inv = np.asarray(my_weight_data.train_y)
X_Y_Spline_inv = make_interp_spline(np.sort(array_train_x_inv), np.sort(unnormalized_log_reg_train_pred_inv))
X_inv = np.linspace(min(np.sort(array_train_x_inv)), max(np.sort(array_train_x_inv)),200) 

Y_inv = X_Y_Spline_inv(X_inv)


plt.scatter(my_weight_data.test_y,unnormalized_log_reg_test_pred_inv,color='red',label='test set',marker='*', s=40)
plt.scatter(my_weight_data.train_y,unnormalized_log_reg_train_pred_inv,color='blue',label='train set',marker='1', s=40)

plt.plot(X_inv, Y_inv,c='green',linewidth=3)

plt.legend(('model','test set','training set'))
plt.title('logistic regression model (weight vs. age)')
plt.xlabel('weight (lb)')
plt.ylabel('age (years)')
plt.show()    


# parity plot
plt.plot(my_weight_data.test_x,unnormalized_log_reg_test_pred_inv,'r.') # test y vs. unnomarlized predictions
plt.plot(my_weight_data.test_x,my_weight_data.test_x,'k-') # diagonal line
plt.title('Parity Plot of Logistic Regression (weight vs age)', size=20)
plt.xlabel('true age', size=14)
plt.ylabel('unnomarlized predictions of age', size=14)
plt.show()


# In[10]:


# unnormalized_log_reg_test_pred_inv = []
# for i in range(len(my_weight_data.normalized_test_y)):
#     this_test_pred_inv = log_reg(my_weight_data.normalized_test_y[i],popt_inv[0],popt_inv[1],popt_inv[2],popt_inv[3])
#     unnormalized_log_reg_test_pred_inv.append(this_test_pred_inv*np.std(my_weight_data.train_x) + np.mean(my_weight_data.train_x))

# unnormalized_log_reg_train_pred_inv = []
# for i in range(len(my_weight_data.normalized_train_y)):
#     this_train_pred_inv = log_reg(my_weight_data.normalized_train_y[i],popt_inv[0],popt_inv[1],popt_inv[2],popt_inv[3])
#     unnormalized_log_reg_train_pred_inv.append(this_train_pred_inv*np.std(my_weight_data.train_x) + np.mean(my_weight_data.train_x))

    
# array_train_x_inv = np.asarray(my_weight_data.train_y)
# X_Y_Spline_inv = make_interp_spline(np.sort(array_train_x_inv), np.sort(unnormalized_log_reg_train_pred_inv))
# X_inv = np.linspace(min(array_train_x_inv), max(array_train_x_inv),200) 

# Y_inv = X_Y_Spline_inv(X_inv)


# plt.scatter(my_weight_data.test_y,unnormalized_log_reg_test_pred_inv,color='red',label='test set',marker='*', s=40)
# plt.scatter(my_weight_data.train_y,unnormalized_log_reg_train_pred_inv,color='blue',label='train set',marker='1', s=40)

# # plt.plot(X_inv, Y_inv,c='green',linewidth=3)

# plt.legend(('model','test set','training set'))
# plt.title('logistic regression model (weight vs. age)')
# plt.xlabel('weight (lb)')
# plt.ylabel('age (years)')
# plt.show()    


# In[ ]:




