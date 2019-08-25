# STEP 1: IMPORT COMMON PYTHON LIBRARIES
import pandas as pd
import numpy as np
import sys
import os

# Matplotlib for plotting figures
import matplotlib as mpl
import matplotlib.pyplot as plt

# Where to save the figures
EXERCISE_ROOT_DIR = "."
IMAGES_PATH = os.path.join(EXERCISE_ROOT_DIR, "images")

# The function allows images to be saved
def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

# Multivariate linear regression class
class MultivariateLinearRegressionGD(object):
    def __init__(self, alpha=0.001, n_iter=20, print_cost=False):
        self.alpha = alpha
        self.n_iter = n_iter
        self.print_cost = print_cost

    def cost_function(self, X, y):
        y_hat = self.hypothesis(X)
        return (1/(2*len(y))) * (np.dot((y_hat-y).T,(y_hat-y)))

    def fit(self, X, y):
        self.w_    = np.zeros(X.shape[1])
        self.cost_ = []
        print("Iterations\t\tCost\n==========\t\t====")
        for i in range(self.n_iter):
            y_hat    = self.hypothesis(X)
            errors   = (y_hat - y)
            self.w_ -= self.alpha * (1/len(y)) * np.dot(X.T,errors)
            cost     = (1/(2*len(y))) * (np.dot((y_hat-y).T,(y_hat-y)))
            self.cost_.append(cost)
            if self.print_cost and i % 10 == 0:
                print("{}\t\t\t{}".format(i,cost),\
                      file=sys.stdout,\
                      flush=True)
        return self

    def hypothesis(self,X):
        return np.dot(X,self.w_)

    def predict(self,X):
        return self.hypothesis(X)

# Normal equation for linear equations
def normalEquation(X, y):
    return np.dot(np.dot(np.linalg.pinv(np.dot(X.T,X)), X.T),y)

# Function to normalize features
def featureNormalize(X):
    # Computing the Mean and STD of X
    mu = np.mean(X, axis=0, keepdims=True);
    sigma = np.std(X, axis = 0, keepdims=True);
    X_norm = (X - mu)/sigma
    return X_norm, mu, sigma

# Function to plot the cost change against # of iterations for MLR 
def plot_mlr_bgd(model):
    g = plt.figure(1)
    plt.plot(range(1, model.n_iter+1), model.cost_, 'g-s', label= r'${J{(\theta)}}$')
    plt.grid(True)
    learning_rate = r"$\alpha = {}$".format(model.alpha)
    plt.title("BGD with " + str(learning_rate) + " learning rate") 
    plt.xlabel('# of Iterations')
    plt.ylabel(r'${J{(\theta)}}$', rotation=1)
    plt.axis([0,50,0,7e10])
    plt.legend()

def main():
    print("\nProgramming Exercise 1: Multivariate Linear Regression")
    # Load dataset for computation
    input("\nPress <ENTER> key to load dataset ...\n")
    data = pd.read_csv(os.getcwd() + '\data\ex1data2.txt', header=None).values
    X = data[:,:2]
    y = data[:,2]
    print("... dataset loaded successfully!")

    # Normalize features
    X_norm, mu, sigma = featureNormalize(X)
    Xtrain = np.insert(X_norm,0,1,axis=1)

    # CREATE AN OBJECT OF 'MultivariateLinearRegression'
    input("\nPress <ENTER> key to run MLR(using BGD) ...")
    mlr = MultivariateLinearRegressionGD(alpha=0.1, n_iter=50, print_cost=True)
    mlr.fit(Xtrain,y)
    print("\nThe minimum point(found by BGD) is \n%s"%(str(mlr.w_)))

    # PLOT THE COST GRAPH
    input("\nPress <ENTER> key to plot the BGD graph ...")
    plt.figure(1)
    plot_mlr_bgd(mlr)
    plt.show(block=False)

    # Testing model
    input("\nPress <ENTER> key to test model ...\n")
    test_x = [[1650, 3]]
    print("Test example:",test_x)
    norm_x = np.divide((test_x - mu),sigma)
    test_x = np.insert(norm_x,0,1,axis=1)
    price  = mlr.predict(test_x)
    print('\nPrice of a 1650 sq-ft, 3 br house (using BGD): $%.2f'%(price))
    input("\nPress <ENTER> key to run regression using Normal Equation ...")
    
    # Computing theta using Normal Equation
    theta = normalEquation(Xtrain, y)
    print("\nThe minimum point(found by NormalEquation) is \n%s"%(str(theta)))
    price = np.dot(test_x,theta)
    print('\nPrice of a 1650 sq-ft, 3 br house (using NEp): $%.2f' %(price))

    # Terminate program
    input("\nPress <ENTER> key to terminate the process ...")

if __name__ == "__main__":
    main()

