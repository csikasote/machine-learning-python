# STEP 1: IMPORT COMMON PYTHON LIBRARIES
import pandas as pd
import numpy as np
import sys
import os

# Matplotlib for plotting figures
import matplotlib as mpl
import matplotlib.pyplot as plt

# IMPORT HELPER FUNCTIONS
from linear_model_utils import featureNormalize
from linear_model_utils import plot_mlr_bgd, save_fig
from linear_model_utils import MultivariateLinearRegressionGD
from linear_model_utils import normalEquation

# STEP 1: IMPORTING DATA FILES
path = os.getcwd() + '\data\ex1data2.txt' #PATH TO THE DATASET
data = pd.read_csv(path, header=None).values
X = data[:,:2]
y = data[:,2]

# NORMALIZE FEATURES
X_norm, mu, sigma = featureNormalize(X)
Xtrain = np.insert(X_norm,0,1,axis=1)


# CREATE AN OBJECT OF 'MultivariateLinearRegression'
input("\nPress <ENTER> key to run regression using BGD ...")
print(" ")
mlr = MultivariateLinearRegressionGD(alpha=0.1, n_iter=50, print_cost=True)
mlr.fit(Xtrain,y)
#print("\nThe minimum point(found by BGD) is %s"%(str(mlr.w_)))

# PLOT THE BGD COST GRAPH
#input("\nPress <ENTER> key to plot the BGD cost graph ...")
#plot_mlr_bgd(mlr)
#save_fig("MLR_COST_GRAPH")
#plt.show()

# TESTING THE MODEL
test_x = [[1650, 3]]
norm_x = np.divide((test_x - mu),sigma)
test_x = np.insert(norm_x,0,1,axis=1)
price  = mlr.predict(test_x)
print('\nPrice of a 1650 sq-ft, 3 br house (using BGD): $%.2f'%(price))

input("\nPress <ENTER> key to run regression using Normal Equation ...")
# Computing theta using Normal Equation
theta = normalEquation(Xtrain, y)
#print("\nThe minimum point(found by NormalEquation) is %s"%(str(theta)))
price = np.dot(test_x,theta)
print('\nPrice of a 1650 sq-ft, 3 br house (using NEp): $%.2f' %(price))
input("\nPress <ENTER> key to terminate the process ...")




