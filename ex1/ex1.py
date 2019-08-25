# Import common python libraries
import numpy as np
import pandas as pd
import os

# Matplotlib for plotting figures
import matplotlib as mpl
import matplotlib.pyplot as plt

# Where to save the figures
EXERCISE_ROOT_DIR = "."
IMAGES_PATH = os.path.join(EXERCISE_ROOT_DIR, "images")

# Simple linear regression class
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


# The function allows images to be saved
def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

# plotData() function to visualize the distribution of training data
def plot_data(X,y):
    plt.plot(X,y,'rx',markersize=10, label='Training Example')
    plt.grid(True)
    plt.ylabel('Profit in $10,000s')
    plt.xlabel('Population of City in 10,000s')
    plt.title('Training data')
    plt.axis([5, 25, -5, 25])
    plt.legend()

# Function to plot the cost change against # of iterations for SLR 
def plot_cost(model):
    plt.plot(range(1, model.n_iter+1), model.cost_, 'b-o', label= r'${J{(\theta)}}$')
    plt.xlabel('# of Iterations')
    plt.ylabel(r'${J{(\theta)}}$', rotation=1)
    plt.xlim([0,model.n_iter])
    plt.ylim([4,7])
    plt.grid(True)
    plt.title('Batch Gradient Descent (BGD)')
    plt.legend()

# Fit SLR
def plot_fit(Xtrain,X,y,model):
    plt.plot(X,y,'rx',markersize=10, label='Training Example')
    plt.plot(Xtrain, model.predict(Xtrain),'--', color='blue', lw=2)
    plt.grid(True) 
    plt.ylabel('Profit in $10,000s')
    plt.xlabel('Population of City in 10,000s')
    titler = 'Linear Model: Y = %.3f + %.3fx1'%(model.w_[0], model.w_[1])
    plt.title(titler)
    plt.axis([5, 25, -5, 25])
    plt.legend()

def main():
    print("\nProgramming Exercise 1: Linear Regression with Single Variable\n")
    # Load dataset for computation
    data = pd.read_csv(os.getcwd() + '\data\ex1data1.txt', header=None).values
    X = data[:,0].reshape(-1,1)
    y = data[:,1]

    # Adding the bias term to the dataset
    input("\nPress <ENTER> key to visualize dataset ...")
    Xtrain = np.insert(X,0,1,axis=1)
    ytrain = y

    # Visualize the dataset
    plt.figure(1)
    plot_data(X,y)
    plt.show(block=False)

    # Instantiating an object of linear regression class
    lr = SimpleLinearRegressionGD(0.01, 1500)

    # Computing initial cost
    input("\nPress <ENTER> to compute initial cost ...")
    print('\nInitial cost computes is %.2f.'%(lr.MSE(Xtrain,ytrain)))

    # Fit the SLR
    input("\nPress <ENTER> key to run SLR with BGD ...")
    lr.fit(Xtrain, ytrain)
    print("\n\nMinimum point(using BGD):", lr.w_)
    print('\nLinear Model: Y = %.3f + %.3fx1'%(lr.w_[0], lr.w_[1]))

    # Plot Cost vs. Iterations graph
    input("\nPress <ENTER> key to cost vs. iteration graph ...")
    plt.figure(2)
    plot_cost(lr)
    plt.show(block=False)

    # Plot of the fitting line on dataset
    input("\nPress <ENTER> key to plot a fit line ...")
    plt.figure(3)
    plot_fit(Xtrain,X,y,lr)
    plt.show(block=False)

    # Predictions based on the computed weights
    print('\nPredictions ... ', end='')
    input("Press <ENTER> key to make prediction ...")
    print('\nPopulation\t\t Profit\n==========\t\t ======')
    print('%.f \t\t\t %.2f'%\
          (35000, lr.predict([1, 3.5]) * 10000))
    print('%.f \t\t\t %.2f\n'%\
          (70000, lr.predict([1, 7.0]) * 10000))

    # Terminate the program
    input("Press <ENTER> key to terminate program ...")

if __name__ == "__main__":
    main()
