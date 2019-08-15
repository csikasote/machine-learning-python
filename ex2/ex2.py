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
    
    # Objective function
    def cost_function(self,theta,X,y):
        y_hat = self.sigmoid(np.dot(X,theta))
        return (-1/len(X)) * (np.dot(y.T,np.log(y_hat)) +\
                              np.dot((1-y).T,np.log(1-y_hat)))

    # Using "fmin()" to optimize
    def fit(self,X,y,theta):
        return opt.fmin(func=self.cost_function,
                        x0=theta,
                        args=(X,y),
                        maxiter=self.n_iter,
                        full_output=True)
    # Predict function
    def predict(self,X,theta):
        probability = self.sigmoid(np.dot(X,theta))
        return [1 if p >= 0.5 else 0 for p in probability]

    # Classification function
    def classify(self,X,theta):  
        probability = self.sigmoid(np.dot(X,theta))
        return ["ADMIT" if p >= 0.5 else "DECLINE" for p in probability]

    # Decision function
    def decision_(self,X,theta):
        return "\nProbability: %2f"%(self.sigmoid(np.dot(X,theta))[0])\
               + ' \nPrediciton: ' + str(self.predict(X,theta)[0]) \
               + ' \nModel Decision: ' + str(self.classify(X,theta)[0]) \

    
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
    print('\nThe minimum point found: ',result[0])
    print('\nComputed cost at minimum point: %.6f\n' %(result[1]))

    # Testing model with predictions
    input("\nPress <ENTER> key to test the model ...")
    Xtest = [[1, 45, 85]] #TEST EXAMPLE
    decision = lr.decision_(Xtest,result[0])
    print(decision)
    input('\nPress <ENTER> to terminate program ...')

if __name__ == "__main__":
    input("\nPress <ENTER> to run program ...")
    main()

    
