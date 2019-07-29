# Import common python libraries
import pandas as pd
import numpy as np
import os

# Matplotlib for plotting figures
import matplotlib as mpl
import matplotlib.pyplot as plt

# Simple Linear Regression class
class SimpleLR(object):
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
file = os.getcwd() + '\data\ex1data1.txt'
data = pd.read_csv(file, header=None).values
X = data[:,0].reshape(-1,1)
X = np.insert(X,0,1,axis=1)
y = data[:,1]


# Running Simple Linear Regression
lr = SimpleLR(0.01, 1500)

# Initial cost
print('Initial computed cost is %.2f'%(lr.MSE(X,y)))

# Fit the SLR
lr.fit(X, y)
print("\nComputed theta(using BGD):", lr.w_)

# Making prediction
print('\nPopulation\t\t Profit\n==========\t\t ======')
print('%.f \t\t\t %.2f'%(35000, lr.predict([1, 3.5]) * 10000))
print('%.f \t\t\t %.2f\n'%(70000, lr.predict([1, 7.0]) * 10000))


# Batch gradient descent convergence plot
def graph():
    plt.plot(range(1, lr.n_iter+1), lr.cost_, 'b-o', label= r'${J{(\theta)}}$')
    plt.xlabel('# of Iterations')
    plt.ylabel(r'${J{(\theta)}}$', rotation=1)
    plt.xlim([0,lr.n_iter])
    plt.ylim([4,7])
    plt.title('Batch Gradient Descent (BGD)')
    plt.legend()
    plt.grid(True)
    plt.show()

graph()
