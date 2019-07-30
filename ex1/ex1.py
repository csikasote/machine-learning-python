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

# Function to pause the execution of the program
def pause():
    pauseProgram = input("Press the <ENTER> key to continue...")

# Load data for computation
def load_data(file):
    data = pd.read_csv(file, header=None).values
    X = data[:,0].reshape(-1,1)
    y = data[:,1]
    return X, y

print('Loading data ... ', end='')
file = os.getcwd() + '\data\ex1data1.txt'
X,y = load_data(file)

# plotData() function to visualize the distribution of training data
input("Press the <ENTER> key to continue...")
f = plt.figure(1)
plt.plot(X,y,'rx',markersize=10, label='Training Example')
plt.grid(True) #Always plot.grid
plt.ylabel('Profit in $10,000s')
plt.xlabel('Population of City in 10,000s')
plt.title('Training data')
plt.axis([5, 25, -5, 25])
plt.legend()

# Adding the bias term to the dataset
Xtrain = np.insert(X,0,1,axis=1)
ytrain = y

# Running Simple Linear Regression
lr = LinearRegressionGD(0.01, 1500)

# Computing initial cost
print('\nInitial cost computes is %.2f.'%(lr.MSE(Xtrain,ytrain)))

# Fit the SLR
print('\nRunning linear regression with BGD ... ', end='')
input("Press the <ENTER> key to continue...")
lr.fit(Xtrain, ytrain)
print("\nComputed theta(using BGD):", lr.w_)

# Making prediction
print('\nPredictions ... ', end='')
input("Press the <ENTER> key to continue...")
print('\nPopulation\t\t Profit\n==========\t\t ======')
print('%.f \t\t\t %.2f'%(35000, lr.predict([1, 3.5]) * 10000))
print('%.f \t\t\t %.2f\n'%(70000, lr.predict([1, 7.0]) * 10000))


# Batch gradient descent convergence plot
input("Press the <ENTER> key to continue...")
g = plt.figure(2)
plt.plot(range(1, lr.n_iter+1), lr.cost_, 'b-o', label= r'${J{(\theta)}}$')
plt.xlabel('# of Iterations')
plt.ylabel(r'${J{(\theta)}}$', rotation=1)
plt.xlim([0,lr.n_iter])
plt.ylim([4,7])
plt.title('Batch Gradient Descent (BGD)')
plt.legend()
plt.grid(True)
plt.show()

