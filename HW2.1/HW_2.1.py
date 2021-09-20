#!/usr/bin/env python
# coding: utf-8


# change the parameters in the "generate_plots()" function at the bottom
# to test different algorithms and batch options

# algo: 'GD' or 'momentum'
# method: 'batch' or 'mini-batch' or 'sgd'




import json
import matplotlib.pyplot as plt
import numpy as np
import os
import random
from random import shuffle




# copy from hw1. data class to read, split and visualize the data

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



# read in the data
my_weight_data = Data(filename='../HW2.1/weight.json')
my_weight_data.read_from_json()
my_weight_data.train_test_split()
# my_weight_data.visualize_data()
my_weight_data.normalize()





# logistic regression function
def log_reg(x,A,b,w,s):
    z= (x-b)/w
    return A/(1+np.exp(-z)) + s


iterations=[]
loss_train=[]
loss_val=[]

iteration=0

# logistic regression loss function
def log_reg_loss(po,x,y):
    global iteration, iterations, loss_train, loss_val

    iterations.append(iteration)
    pred = []
    try:
        for i in range(len(x)):
            this_pred = log_reg(x[i],po[0],po[1],po[2],po[3])
            pred.append(this_pred)
        log_reg_mse = np.square(np.subtract(y,pred)).mean()
        loss_train.append(log_reg_mse)
        iteration+=1
        return log_reg_mse
    except TypeError:
        this_pred = log_reg(x,po[0],po[1],po[2],po[3])
        pred.append(this_pred)
        log_reg_mse = np.square(np.subtract(y,pred)).mean()
        loss_train.append(log_reg_mse)
        iteration+=1
        return log_reg_mse

# helper function that returns loss of log_reg function with p(initial guess of parameter)
# and x (training data) as input
def objective_fun(p,X_y_data):

    return log_reg_loss(p,X_y_data[0], X_y_data[1]) # return loss of the log_reg function


# get train, test data
X_y_norm_train_data = (my_weight_data.normalized_train_x,my_weight_data.normalized_train_y)
X_y_norm_test_data = (my_weight_data.normalized_test_x,my_weight_data.normalized_test_y)
X_y_array=np.stack((my_weight_data.normalized_train_x,my_weight_data.normalized_train_y),axis=1)
# PARAM
xmin = -5
xmax = 5





def optimizer(objective, NDIM, algo='GD', LR=0.05, method='batch',decay_factor=0.9):
    xi=np.random.uniform(xmin,xmax,NDIM)

    print("INITAL GUESS: ",xi)
    dx=0.0001  #STEP SIZE FOR FINITE DIFFERENCE
    t=0        #INITIAL ITERATION COUNTER
    tmax=100000  #MAX NUMBER OF ITERATION
    tol=10**-10   #EXIT AFTER CHANGE IN F IS LESS THAN THIS



    # algo
    if algo == 'GD':
        if method == 'batch':
            train_cost_all_iter = []
            test_cost_all_iter = []
            while(t<=tmax):
                t=t+1
                #NUMERICALLY COMPUTE GRADIENT (finite difference)
                df_dx=np.zeros(NDIM) #parameter
                for i in range(0,NDIM):
                    dX=np.zeros(NDIM);
                    dX[i]=dx;
                    xm1=xi-dX;
                    df_dx[i]=(objective(xi,X_y_norm_train_data)-objective(xm1,X_y_norm_train_data))/dx

                xip1=xi-LR*df_dx #STEP
                cost_this_iter = objective(xi,X_y_norm_train_data)
                test_cost_this_iter = objective(xi,X_y_norm_test_data)
                if(t%1000==0):
                    df=np.mean(np.absolute(objective(xip1,X_y_norm_train_data)-objective(xi,X_y_norm_train_data)))
                    print(t,"	",xi,"	","	",objective(xi,X_y_norm_train_data))

                    if(df<tol):
                        print("STOPPING CRITERION MET (STOPPING TRAINING)")
                        break

                #UPDATE FOR NEXT ITERATION OF LOOP
                xi=xip1
                train_cost_all_iter.append(cost_this_iter)
                test_cost_all_iter.append(test_cost_this_iter)
            return xi, train_cost_all_iter, test_cost_all_iter, t


        if method == 'mini-batch':
            avg_cost_all_iters = []
            test_cost_all_iters = []
            while(t<=tmax):
                t=t+1
                # create mini batches
                # mini batch size = 0.5 batch
                cost_this_iter = [] #2
                test_cost_this_iter = []
                num_mini_batch = int(len(X_y_norm_train_data[0])/(0.5*len(X_y_norm_train_data[0])))
                batch_size = int(0.5*len(X_y_norm_train_data[0]))
                mini_batches = []
                for j in range(num_mini_batch):
                    mini_batches.append((X_y_norm_train_data[0][j*batch_size:(j+1)*batch_size],X_y_norm_train_data[1][j*batch_size:(j+1)*batch_size]))
                    cost_this_iter_this_batch = objective(xi,mini_batches[j])
                    test_cost_this_iter_this_batch = objective(xi,X_y_norm_test_data)
                    #NUMERICALLY COMPUTE GRADIENT (finite difference)
                    df_dx = np.zeros(NDIM)
                    for i in range(0,NDIM):
                        dX=np.zeros(NDIM);
                        dX[i]=dx;
                        xm1=xi-dX;
                        df_dx[i]=(objective(xi,mini_batches[j])-objective(xm1,mini_batches[j]))/dx

                    cost_this_iter.append(cost_this_iter_this_batch)
                    test_cost_this_iter.append(test_cost_this_iter_this_batch)
                    xip1=xi-LR*df_dx #STEP

                    if(t%1000==0):
                        df=np.mean(np.absolute(objective(xip1,mini_batches[j])-objective(xi,mini_batches[j])))
                        print(t,"	",xi,"	","	",objective(xi,mini_batches[j])) #,df)

                        if(df<tol):
                            print("STOPPING CRITERION MET (STOPPING TRAINING)")
                            break

                    #UPDATE FOR NEXT ITERATION OF LOOP

                    xi=xip1
                avg_cost_this_iter = np.mean(cost_this_iter)
                avg_cost_all_iters.append(avg_cost_this_iter)
                avg_test_cost_this_iter = np.mean(test_cost_this_iter)
                test_cost_all_iters.append(avg_test_cost_this_iter)

            return xi, avg_cost_all_iters, test_cost_all_iters, t

        if method == 'sgd':
            avg_cost_all_iters = []
            test_cost_all_iters = []
            while(t<=tmax):
                t=t+1
                # create mini batches
                # mini batch size = 0.5 batch

                cost_this_iter = [] # 200
                test_cost_this_iter = []
                num_mini_batch = len(X_y_norm_train_data[0])
                each_batch = []
                for j in range(num_mini_batch):

                    each_batch.append((X_y_norm_train_data[0][j],X_y_norm_train_data[1][j]))
                    #NUMERICALLY COMPUTE GRADIENT (finite difference)
                    df_dx=np.zeros(NDIM)

                    cost_this_iter_this_batch = objective(xi,each_batch[j])
                    test_cost_this_iter_this_batch = objective(xi,X_y_norm_test_data)
                    for i in range(0,NDIM):
                        dX=np.zeros(NDIM);
                        dX[i]=dx;
                        xm1=xi-dX;
                        df_dx[i]=(objective(xi,each_batch[j])-objective(xm1,each_batch[j]))/dx


                    cost_this_iter.append(cost_this_iter_this_batch)
                    test_cost_this_iter.append(test_cost_this_iter_this_batch)
                    xip1=xi-LR*df_dx #STEP

                    if(t%1000==0):
                        df=np.mean(np.absolute(objective(xip1,each_batch[j])-objective(xi,each_batch[j])))
                        print(t,"	",xi,"	","	",objective(xi,each_batch[j])) #,df)

                        if(df<tol):
                            print("STOPPING CRITERION MET (STOPPING TRAINING)")
                            break

                    #UPDATE FOR NEXT ITERATION OF LOOP
                    xi=xip1
                avg_cost_this_iter = np.mean(cost_this_iter)
                avg_cost_all_iters.append(avg_cost_this_iter)
                avg_test_cost_this_iter = np.mean(test_cost_this_iter)
                test_cost_all_iters.append(avg_test_cost_this_iter)
            return xi, avg_cost_all_iters, test_cost_all_iters, t




    # ---------------------------- GD + momentum ------------------------------- #
    # -------------------------------------------------------------------------- #
    # -------------------------------------------------------------------------- #

    if algo == 'momentum':
        delta_model_parameters = np.zeros(NDIM)
        if method == 'batch':
            train_cost_all_iter = []
            test_cost_all_iter = []
            while(t<=tmax):
                t=t+1

                #NUMERICALLY COMPUTE GRADIENT (finite difference)
                df_dx=np.zeros(NDIM) #parameter
                for i in range(0,NDIM):
                    dX=np.zeros(NDIM);
                    dX[i]=dx;
                    xm1=xi-dX;
                    df_dx[i]=(objective(xi,X_y_norm_train_data) - objective(xm1,X_y_norm_train_data))/dx

                xip1=xi-LR*df_dx #STEP
                cost_this_iter = objective(xi,X_y_norm_train_data)
                test_cost_this_iter = objective(xi,X_y_norm_test_data)

                if(t%1000==0):
                    df=np.mean(np.absolute(objective(xip1,X_y_norm_train_data)-objective(xi,X_y_norm_train_data)))
                    print(t,"	",xi,"	","	",objective(xi,X_y_norm_train_data))
#                     print(delta_model_parameters)
#                     print(xip1)


                    if(df<tol):
                        print("STOPPING CRITERION MET (STOPPING TRAINING)")
                        break

                #UPDATE FOR NEXT ITERATION OF LOOP

                xi = xip1 - decay_factor * delta_model_parameters
                delta_model_parameters = LR * df_dx
                train_cost_all_iter.append(cost_this_iter)
                test_cost_all_iter.append(test_cost_this_iter)
            return xi, train_cost_all_iter, test_cost_all_iter, t

        if method == 'mini-batch':
            avg_cost_all_iters = []
            test_cost_all_iters = []
            while(t<=tmax):
                t=t+1
                cost_this_iter = []
                test_cost_this_iter = []
                # create mini batches
                # mini batch size = 0.5 batch
                num_mini_batch = int(len(X_y_norm_train_data[0])/(0.5*len(X_y_norm_train_data[0])))
                batch_size = int(0.5*len(X_y_norm_train_data[0]))
                mini_batches = []
                for j in range(num_mini_batch):
                    mini_batches.append((X_y_norm_train_data[0][j*batch_size:(j+1)*batch_size],X_y_norm_train_data[1][j*batch_size:(j+1)*batch_size]))

                    cost_this_iter_this_batch = objective(xi,mini_batches[j])
                    test_cost_this_iter_this_batch = objective(xi,X_y_norm_test_data)
                    #NUMERICALLY COMPUTE GRADIENT (finite difference)
                    df_dx=np.zeros(NDIM)
                    for i in range(0,NDIM):
                        dX=np.zeros(NDIM);
                        dX[i]=dx;
                        xm1=xi-dX;
                        df_dx[i]=(objective(xi,mini_batches[j])-objective(xm1,mini_batches[j]))/dx

                    cost_this_iter.append(cost_this_iter_this_batch)
                    test_cost_this_iter.append(test_cost_this_iter_this_batch)
                    xip1=xi-LR*df_dx #STEP

                    if(t%1000==0):
                        df=np.mean(np.absolute(objective(xip1,mini_batches[j])-objective(xi,mini_batches[j])))
                        print(t,"	",xi,"	","	",objective(xi,mini_batches[j])) #,df)

                        if(df<tol):
                            print("STOPPING CRITERION MET (STOPPING TRAINING)")
                            break

                    #UPDATE FOR NEXT ITERATION OF LOOP
                    xi = xip1 - decay_factor * delta_model_parameters
                    delta_model_parameters = LR * df_dx
                avg_cost_this_iter = np.mean(cost_this_iter)
                avg_cost_all_iters.append(avg_cost_this_iter)
                avg_test_cost_this_iter = np.mean(test_cost_this_iter)
                test_cost_all_iters.append(avg_test_cost_this_iter)
            return xi, avg_cost_all_iters, test_cost_all_iters, t

        if method == 'sgd':
            avg_cost_all_iters = []
            test_cost_all_iters = []
            while(t<=tmax):
                t=t+1

                cost_this_iter = [] # 200
                test_cost_this_iter = []
                num_mini_batch = len(X_y_norm_train_data[0])
                each_batch = []
                for j in range(num_mini_batch):

                    each_batch.append((X_y_norm_train_data[0][j],X_y_norm_train_data[1][j]))
                    #NUMERICALLY COMPUTE GRADIENT (finite difference)
                    df_dx=np.zeros(NDIM)

                    cost_this_iter_this_batch = objective(xi,each_batch[j])
                    test_cost_this_iter_this_batch = objective(xi,X_y_norm_test_data)

                    for i in range(0,NDIM):
                        dX=np.zeros(NDIM);
                        dX[i]=dx;
                        xm1=xi-dX;
                        df_dx[i]=(objective(xi,each_batch[j])-objective(xm1,each_batch[j]))/dx


                    cost_this_iter.append(cost_this_iter_this_batch)
                    test_cost_this_iter.append(test_cost_this_iter_this_batch)

                    xip1=xi-LR*df_dx #STEP

                    if(t%1000==0):
                        df=np.mean(np.absolute(objective(xip1,each_batch[j])-objective(xi,each_batch[j])))
                        print(t,"	",xi,"	","	",objective(xi,each_batch[j])) #,df)

                        if(df<tol):
                            print("STOPPING CRITERION MET (STOPPING TRAINING)")
                            break

                    #UPDATE FOR NEXT ITERATION OF LOOP
                    xi = xip1 - decay_factor * delta_model_parameters
                    delta_model_parameters = LR * df_dx
                avg_cost_this_iter = np.mean(cost_this_iter)
                avg_cost_all_iters.append(avg_cost_this_iter)
                avg_test_cost_this_iter = np.mean(test_cost_this_iter)
                test_cost_all_iters.append(avg_test_cost_this_iter)
            return xi, avg_cost_all_iters,test_cost_all_iters, t




# helper function to train, test and plot

def generate_plots(objective, NDIM, algo='GD', LR= 0.05, method = 'batch', decay_factor=0.9):
    popt, train_loss, test_loss, total_iter_num =  optimizer(objective_fun, NDIM, algo, LR, method, decay_factor)

    # predictions: GD with batch
    unnormalized_log_reg_train_pred = []
    for i in range(len(my_weight_data.normalized_train_x)):
        this_train_pred = log_reg(my_weight_data.normalized_train_x[i],popt[0],popt[1],popt[2],popt[3])
        unnormalized_log_reg_train_pred.append(this_train_pred*np.std(my_weight_data.train_y) + np.mean(my_weight_data.train_y))

    unnormalized_log_reg_test_pred = []
    for i in range(len(my_weight_data.normalized_test_x)):
        this_test_pred = log_reg(my_weight_data.normalized_test_x[i],popt[0],popt[1],popt[2],popt[3])
        unnormalized_log_reg_test_pred.append(this_test_pred*np.std(my_weight_data.train_y) + np.mean(my_weight_data.train_y))

    # function plots
    fig, ax = plt.subplots()
    ax.plot(my_weight_data.train_x, my_weight_data.train_y, 'o', label='Training set')
    ax.plot(my_weight_data.test_x, my_weight_data.test_y, 'x', label='Test set')
    ax.plot(sorted(my_weight_data.train_x),sorted(unnormalized_log_reg_train_pred), '-', label='Model')
    plt.xlabel('x', fontsize=18)
    plt.ylabel('y', fontsize=18)
    plt.legend()
    plt.title('model trained')
    plt.show()

    # parity plot
    fig, ax = plt.subplots()
    ax.plot(unnormalized_log_reg_train_pred, my_weight_data.train_y, 'o', label='Training set')
    ax.plot(unnormalized_log_reg_test_pred, my_weight_data.test_y, 'o', label='Test set')
    # ax.plot(yt, yt, '-', label='y_pred=y_data')

    plt.xlabel('y predicted', fontsize=18)
    plt.ylabel('y data', fontsize=18)
    plt.legend()
    plt.show()

    #MONITOR TRAINING AND VALIDATION LOSS

    fig, ax = plt.subplots()
    #iterations,loss_train,loss_val
    # test_loss = log_reg_loss(popt_sgd_mom,my_weight_data.test_x,my_weight_data.test_y)
    iter_list = list(range(0, total_iter_num-1))
    ax.plot(list(range(0, len(train_loss))),train_loss, 'o', label='Training loss')
    ax.plot(list(range(0, len(test_loss))),test_loss,'o',label='Testing loss')
    plt.xlabel('optimizer iterations', fontsize=18)
    plt.ylabel('loss', fontsize=18)
    plt.legend()
    plt.show()





# change the parameters in the "generate_plots()" function below
# to test different algorithms and batch options

# algo: 'GD' or 'momentum'
# method: 'batch' or 'mini-batch' or 'sgd'

generate_plots(objective_fun,4,method='batch',algo='GD')
#generate_plots(objective_fun,4,method='mini-batch',algo='GD')
#generate_plots(objective_fun,4,method='sgd',algo='GD')
#generate_plots(objective_fun,4,method='batch',algo='momentum')
#generate_plots(objective_fun,4,method='mini-batch',algo='momentum')
#generate_plots(objective_fun,4,method='sgd',algo='momentum')
