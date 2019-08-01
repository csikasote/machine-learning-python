# Import common python libraries
import numpy as np
import os

# Matplotlib for plotting figures
import matplotlib as mpl
import matplotlib.pyplot as plt

# Import helper function
from linear_model_utils import load_data, plot_data, save_fig
from linear_model_utils import plot_cost, bgd_path, plot_fit
from linear_model_utils import SimpleLinearRegressionGD

# Load data for computation
print('Loading data ... ', end='')
file = os.getcwd() + '\data\ex1data1.txt'
X,y = load_data(file)

# Adding the bias term to the dataset
input("Press <ENTER> key to continue ..")
Xtrain = np.insert(X,0,1,axis=1)
ytrain = y

# Creating an object of linear regression
lr = SimpleLinearRegressionGD(0.01, 1500)

# Computing initial cost
print('\nInitial cost computes is %.2f.'%(lr.MSE(Xtrain,ytrain)))

# Fit the SLR
print('\nRunning linear regression with BGD ... ', end='')
input("")
lr.fit(Xtrain, ytrain)
print("\nMinimum point(using BGD):", lr.w_)
print('\nLinear Model: Y = %.3f + %.3fx1'%(lr.w_[0], lr.w_[1]))

# Making prediction
print('\nPredictions ... ', end='')
input("Press <ENTER> key to continue ...")
print('\nPopulation\t\t Profit\n==========\t\t ======')
print('%.f \t\t\t %.2f'%(35000, lr.predict([1, 3.5]) * 10000))
print('%.f \t\t\t %.2f\n'%(70000, lr.predict([1, 7.0]) * 10000))

# Plot training data
input("Press <ENTER> key to plot and save the figures ...")
plot_data(X,y)
save_fig("training_data")
plot_cost(lr)
save_fig("BGD")
plot_fit(Xtrain,X,y,lr)
save_fig("fit_line")
plt.show()
input("Press <ENTER> key to continue ...")
