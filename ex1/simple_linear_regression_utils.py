# Import common python libraries
import pandas as pd
import numpy as np
import system
import os

# Matplotlib for plotting figures
import matplotlib as mpl
import matplotlib.pyplot as plt

# Simple Linear Regression class
class LinearRegressionGD(object):
    def __init__(self, alpha=0.001, n_iter=20):
        self.alpha    = alpha
        self.n_iter   = n_iter
    
    def MSE(self, X, y):
        self.w_  = np.zeros(X.shape[1])
        y_hat    = self.hypothesis(X)
        errors   = (y_hat - y)
        squared_errors = errors**2
        return (1/(2*len(y))) * squared_errors.sum()

    def fit(self, X, y):
        self.w_    = np.zeros(X.shape[1])
        self.cost_ = []
        for i in range(self.n_iter):
            y_hat    = self.hypothesis(X)
            errors   = (y_hat - y)
            gradient = self.alpha * np.dot(X.T, errors)
            self.w_  -= self.alpha * gradient
            cost     = np.sum(errors**2) / (len(y)*2.0)
            self.cost_.append(cost)
        return self
        
    def hypothesis(self, X):
        return np.dot(X,self.w_)

    def predict(self, X):
        return self.hypothesis(X)

# Load data for computation
def load_data(file):
    data = pd.read_csv(file, header=None).values
    X = data[:,0].reshape(-1,1)
    y = data[:,1]
    return X, y

# plotData() function to visualize the distribution of training data
def plot_data(X,y):
    f = plt.figure(1)
    plt.plot(X,y,'rx',markersize=10, label='Training Example')
    plt.grid(True)
    plt.ylabel('Profit in $10,000s')
    plt.xlabel('Population of City in 10,000s')
    plt.title('Training data')
    plt.axis([5, 25, -5, 25])
    plt.legend()

# Batch gradient descent convergence plot
def plot_BGD(model):
    g = plt.figure(2)
    plt.plot(range(1, model.n_iter+1), model.cost_, 'b-o', label= r'${J{(\theta)}}$')
    plt.xlabel('# of Iterations')
    plt.ylabel(r'${J{(\theta)}}$', rotation=1)
    plt.xlim([0,model.n_iter])
    plt.ylim([4,7])
    plt.grid(True)
    plt.title('Batch Gradient Descent (BGD)')
    plt.legend()

# Fitting the linear model
def plotFit(Xtrain,X,y,model):
    h = plt.figure(3)
    plt.plot(X,y,'rx',markersize=10, label='Training Example')
    plt.plot(Xtrain, model.predict(Xtrain), color='black', lw=2)
    plt.grid(True) 
    plt.ylabel('Profit in $10,000s')
    plt.xlabel('Population of City in 10,000s')
    titler = 'Linear Model: Y = %.3f + %.3fx1'%(model.w_[0], model.w_[1])
    plt.title(titler)
    plt.axis([5, 25, -5, 25])
    plt.legend()
