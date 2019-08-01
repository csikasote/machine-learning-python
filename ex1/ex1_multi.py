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
from linear_model_utils import MultivariateLinearRegressionGD
from linear_model_utils import plot_mlr_bgd, save_fig

# STEP 1: IMPORTING DATA FILES
path = os.getcwd() + '\data\ex1data2.txt' #PATH TO THE DATASET
data = pd.read_csv(path, header=None).values
X = data[:,:2]
y = data[:,2]

# NORMALIZE FEATURES
X_norm, mu, sigma = featureNormalize(X)
Xtrain = np.insert(X_norm,0,1,axis=1)


# CREATE AN OBJECT OF 'MultivariateLinearRegression'
input("Press <ENTER> key to continue ...")
print(" ")
mlr = MultivariateLinearRegressionGD(alpha=0.1, n_iter=50, print_cost=True)
mlr.fit(Xtrain,y)
print("\nThe minimum point(found by BGD) is %s"%(str(mlr.w_)))
print("\nCost computed(by BGD) is %s"%(str(mlr.cost_[len(mlr.cost_)-1])))

# PLOT THE BGD COST GRAPH
input("\nPress <ENTER> key to plot the BGD cost graph ...")
plot_mlr_bgd(mlr)
#save_fig("MLR_COST_GRAPH")
plt.show()

# TESTING THE MODEL
input("\nPress <ENTER> key to make prediction ...")
#NORMALIZING TEST X
test_x = [[1650, 3]] # 1650 sq-ft & 3 br house
norm_x = np.divide((test_x - mu),sigma)
test_x = np.insert(norm_x,0,1,axis=1)
print('\nNormalized Test X [[1650, 3]]: ' + str(test_x) + '\n')
price = mlr.predict(test_x)
print('Price of a 1650 sq-ft, 3 br house (using gradient descent): $%.2f'%(price))
input("\nPress <ENTER> key to terminate the process ...")






