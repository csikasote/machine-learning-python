# Common imports 
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# Loading data using pandas
file = os.getcwd() + '\data\ex1data1.txt'
data = pd.read_csv(file, header=None).values
X = data[:,0].reshape(-1,1)
y = data[:,1].reshape(-1,1)

def plot_data(X,y):
    plt.plot(X,y,'rx',markersize=10,
             label='Training Example')
    plt.grid(True)
    plt.ylabel('Profit in $10,000s')
    plt.xlabel('Population of City in 10,000s')
    plt.title('Training data')
    plt.axis([5, 25, -5, 25])
    plt.legend()
    plt.show()

#plot_data(X,y)
def hypothesis(X,theta):
    return np.dot(X,theta)

#SETTINGS FOR RUNNING THE GRADIENT DESCENT ALGORITHM
Xtrain = np.insert(X,0,1,axis=1)
init_theta = np.zeros((2,1))

# COST FUNCTION 
def cost_function(x, y, theta):
    y_hat = hypothesis(x, theta);
    errors      = np.subtract(y_hat,y);
    sqErrors    = np.square(errors);
    return (1/(2*len(y))) * np.sum(sqErrors)

# TESTING COST FUNCTION
#print(cost_function(Xtrain,y,init_theta))

# BATCH GRADIENT DESCENT ALGORITHM
def BGD(X,y,theta, alpha, n_iters):
    for i in range(n_iters):
        y_hat = hypothesis(X,theta) # (m,n).(n,1) = (m,1)
        errors = (y_hat-y) # (m,1)
        gradient = (1/len(y)) * np.dot(X.T,errors) # (n,m).(m,1) = (n,1)
        theta = theta - (alpha * gradient) # (n,1) = (n,1) - (n,1)
    return theta

# TESTING BGD
theta = BGD(Xtrain,y,init_theta,0.01,1500)
print(theta, str(theta.shape))

   
        
    
    
    
