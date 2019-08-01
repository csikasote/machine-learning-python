# Import common python libraries
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

# SIMPLE LINEAR REGRESSION CLASS
class SimpleLinearRegressionGD(object):
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

# Function to normalize features
def featureNormalize(X):
    # Computing the Mean and STD of X
    mu = np.mean(X, axis=0, keepdims=True);
    sigma = np.std(X, axis = 0, keepdims=True);
    # Normalize X 
    X_norm = (X - mu)/sigma
    return X_norm, mu, sigma

# The function allows images to be saved
def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

# Load data for computation
def load_data(file):
    data = pd.read_csv(file, header=None).values
    X = data[:,0].reshape(-1,1)
    y = data[:,1]
    return X, y

# plotData() function to visualize the distribution of training data
def plot_data(X,y):
    f = plt.figure(1)
    plt.plot(X,y,'rx',markersize=10, label='Training Example')
    plt.grid(True)
    plt.ylabel('Profit in $10,000s')
    plt.xlabel('Population of City in 10,000s')
    plt.title('Training data')
    plt.axis([5, 25, -5, 25])
    plt.legend()

# Function to plot the cost change against # of iterations for SLR 
def plot_cost(model):
    g = plt.figure(2)
    plt.plot(range(1, model.n_iter+1), model.cost_, 'b-o', label= r'${J{(\theta)}}$')
    plt.xlabel('# of Iterations')
    plt.ylabel(r'${J{(\theta)}}$', rotation=1)
    plt.xlim([0,model.n_iter])
    plt.ylim([4,7])
    plt.grid(True)
    plt.title('Batch Gradient Descent (BGD)')
    plt.legend()

# Fitting the simple linear model
def plot_fit(Xtrain,X,y,model):
    h = plt.figure(3)
    plt.plot(X,y,'rx',markersize=10, label='Training Example')
    plt.plot(Xtrain, model.predict(Xtrain),'--', color='blue', lw=2)
    plt.grid(True) 
    plt.ylabel('Profit in $10,000s')
    plt.xlabel('Population of City in 10,000s')
    titler = 'Linear Model: Y = %.3f + %.3fx1'%(model.w_[0], model.w_[1])
    plt.title(titler)
    plt.axis([5, 25, -5, 25])
    plt.legend()

# Function to plot the BGD path for simple linear regression
def bgd_path(theta):
    k = plt.figure(4)
    plt.plot(theta[:, 0], theta[:, 1], "m-s", linewidth=3, label="Batch")
    plt.title('Batch gradient descent path plot')
    plt.legend(loc="lower left")
    plt.xlabel(r"$\theta_0$", fontsize=17)
    plt.ylabel(r"$\theta_1$", fontsize=17, rotation=0)
    plt.axis([-4.0, 1.0, -0.2, 1.2])
    plt.grid(True)

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
