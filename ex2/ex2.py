# Import common python libraries
from scipy import optimize as opt
import pandas as pd
import numpy as np
import sys
import os

# Matplotlib for plotting figures
import matplotlib as mpl
import matplotlib.pyplot as plt

# Defining a class Logistic Regression that implements ex2
class LogisticRegression(object):
    def __init__(self, n_iter=500):
        self.n_iter   = n_iter

    # Sigmoid function
    def sigmoid(self, z):
        return (1/(1+ np.exp(-z)))
    
    # Logistic regression objective function
    def cost_function(self,theta,X,y):
        y_hat = self.sigmoid(np.dot(X,theta))
        return (-1/len(X)) * (np.dot(y.T,np.log(y_hat)) +\
                              np.dot((1-y).T,np.log(1-y_hat)))

    # Function to FIT the algorithm
    def fit(self,X,y,theta):
        return opt.fmin(func=self.cost_function,
                        x0=theta,
                        args=(X,y),
                        maxiter=self.n_iter,
                        full_output=True)

    # Prediction function
    def predict(self,X,theta):
        probability = self.sigmoid(np.dot(X,theta))
        return [1 if x >= 0.5 else 0 for x in probability]
    
def main():
    # Load dataset
    path = os.getcwd() + '\data\ex2data1.txt'
    data=pd.read_csv(path, header=None, names=['Exam 1', 'Exam 2', 'Admitted']).values
    X = data[:,[0,1]] 
    y = data[:,[2]]
    
    # Add a column of ones to x for x0 = 1
    X = np.c_[np.ones((len(X), 1)), X]
    init_theta = np.zeros((X.shape[1],1))

    # Instantiate an object the LogisticRegression class
    lr = LogisticRegression()
    # Print the initial cost
    print('\nInitial Cost: %f\n' % (lr.cost_function(init_theta,X,y)))

    # Optimizing the cost function
    result = lr.fit(X,y,init_theta)
    theta = result[0]
    cost = result[1]
    print('\nThe minimum point found(with fmin()): ',theta )
    print('\nComputed cost at minimum point: %.6f' %(cost))

if __name__ == "__main__":
    main()

    
