# Import common python libraries
import pandas as pd
import numpy as np
import sys
import os

# Matplotlib for plotting figures
import matplotlib as mpl
import matplotlib.pyplot as plt


# MULTIVARIATE LINEAR REGRESSION
class MultivariateLinearRegression(object):
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


# FEATURE NORMALIZATION FUNCTION
def featureNormalize(X):
    # Computing the Mean and STD of X
    mu = np.mean(X, axis=0, keepdims=True);
    sigma = np.std(X, axis = 0, keepdims=True);
    # Normalize X 
    X_norm = (X - mu)/sigma
    return X_norm, mu, sigma


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

# DEFINING THE COST CONVERGENCE PLOT
def plot_bgd(model):
    g = plt.figure(1)
    plt.plot(range(1, model.n_iter+1), model.cost_, 'g-s', label= r'${J{(\theta)}}$')
    plt.grid(True)
    learning_rate = r"$\alpha = {}$".format(model.alpha)
    plt.title("BGD with " + str(learning_rate) + " learning rate") 
    plt.xlabel('# of Iterations')
    plt.ylabel(r'${J{(\theta)}}$', rotation=1)
    plt.axis([0,50,0,7e10])
    plt.legend()
