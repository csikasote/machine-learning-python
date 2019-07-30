# Import common python libraries
import numpy as np
import os

# Matplotlib for plotting figures
import matplotlib as mpl
import matplotlib.pyplot as plt

# Import helper function
from simple_linear_regression_utils import load_data, plot_data
from simple_linear_regression_utils import plot_BGD, plotFit
from simple_linear_regression_utils import LinearRegressionGD

# Load data for computation
print('Loading data ... ', end='')
file = os.getcwd() + '\data\ex1data1.txt'
X,y = load_data(file)

# Adding the bias term to the dataset
input("")
Xtrain = np.insert(X,0,1,axis=1)
ytrain = y

# Creating an object of linear regression
lr = LinearRegressionGD(0.01, 1500)

# Computing initial cost
print('\nInitial cost computes is %.2f.'%(lr.MSE(Xtrain,ytrain)))

# Fit the SLR
print('\nRunning linear regression with BGD ... ', end='')
input("")
lr.fit(Xtrain, ytrain)
print("\nComputed theta(using BGD):", lr.w_)
print('\nLinear Model: Y = %.3f + %.3fx1'%(lr.w_[0], lr.w_[1]))

# Making prediction
print('\nPredictions ... ', end='')
input("")
print('\nPopulation\t\t Profit\n==========\t\t ======')
print('%.f \t\t\t %.2f'%(35000, lr.predict([1, 3.5]) * 10000))
print('%.f \t\t\t %.2f\n'%(70000, lr.predict([1, 7.0]) * 10000))

# Plot training data
plot_data(X,y)
plot_BGD(lr)
plotFit(Xtrain,X,y,lr)
plt.show()
input("Press <ENTER> key to continue ...")
