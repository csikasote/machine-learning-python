# Import common python libraries
from scipy import optimize as opt
import pandas as pd
import numpy as np
import os

# Matplotlib for plotting figures
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


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
    
# Logistic regression with regularization
class LogisticRegression(object):
    def __init__(self, n_iter=500, Lambda=0.001):
        self.n_iter = n_iter

    # Sigmoid function
    def sigmoid(self, z):
        return (1/(1+ np.exp(-z)))

    # Net Input
    def net_input(self,X,theta):
        return np.dot(X,theta)
    
    # Objective function with regulization term
    def cost_function(self,theta,X, y,Lambda):
        z = self.net_input(X,theta)
        y_hat = self.sigmoid(z)
        cost = ((-1/len(y)) * (np.dot(y.T,np.log(y_hat)) + \
                               np.dot((1-y).T,np.log(1-y_hat))))
        reg_term = (Lambda/(2*len(y))) * np.sum(np.dot(theta[1:].T,theta[1:]))
        return cost + reg_term
    

    # Using "minimize()" to optimize
    def fit(self,theta,X,y,Lambda):
        args = (X,y,Lambda)
        method = 'BFGS'
        options = {'disp': True, 'maxiter': self.n_iter}
        return opt.minimize(self.cost_function,
                            x0=theta,
                            args=args,
                            method=method,
                            options=options)

    # Function for making predictions
    def predict(self,X,theta):
        z = self.net_input(X,theta)
        probability = self.sigmoid(z)
        return [1 if p >= 0.5 else 0 for p in probability]
    
# The function to transform X features to polynomial features
def mapFeature(x1,x2):
    degree = 6
    out = np.ones((x1.shape[0]))
    for i in range(1,degree+1):  
        for j in range(0,i+1):
            tmp = np.multiply(np.power(x1, i-j),np.power(x2, j))
            out = np.vstack((out,tmp))
    return out.T

# Function to plot the data points
def plot_data(x1,x2,y1,y2): 
    plt.scatter(x1,y1, s=50, c='b', marker='o', label='Admitted')  
    plt.scatter(x2,y2, s=50, c='r', marker='x', label='Not Admitted')   
    plt.xlabel('Microchip Test 1')  
    plt.ylabel('Microchip Test 2')
    plt.title('Scatter plot of Trainig data')
    plt.axis([-1,1.5,-1,1.5])
    plt.grid(True)
    plt.legend() 


# Function to plot the decision function 
def plot_decision_boundary(theta,Lambda,datapoints):
    xvals = np.linspace(-1, 1.5, 50)
    yvals = np.linspace(-1, 1.5, 50)
    zvals = np.zeros((len(xvals), len(yvals)));
    
    for i in range(len(xvals)):
        for j in range(len(yvals)):
            mf = mapFeature(np.array([xvals[i]]), np.array([yvals[j]]));
            zvals[i][j] = np.dot(mf, theta);
    zvals = zvals.T
    plt.scatter(datapoints['Ex1_pos'],datapoints['Ex2_pos'], s=50, c='b', marker='o', label='Admitted')  
    plt.scatter(datapoints['Ex1_neg'],datapoints['Ex2_neg'], s=50, c='r', marker='x', label='Not Admitted')   
    plt.grid(True)
    cs =plt.contour(xvals,yvals, zvals, [0])
    plt.title('Decison Boundary(%s = %d)' % (r'$\lambda$', Lambda))
    plt.legend()


def main():
    path = os.getcwd() + '\data\ex2data2.txt'
    df = pd.read_csv(path, header=None, names=['Test_1', 'Test_2', 'Accepted'])
    pos_df = df[df['Accepted'].isin([1])]
    neg_df = df[df['Accepted'].isin([0])]

    # Data dictionary
    datapoints = {'Ex1_pos':pos_df['Test_1'],
                  'Ex2_pos':pos_df['Test_2'],
                  'Ex1_neg':neg_df['Test_1'],
                  'Ex2_neg':neg_df['Test_2']}

    # Visualize datapoints
    input("\nPress <ENTER> to visualize the data points ...")
    plot_data(datapoints['Ex1_pos'],
              datapoints['Ex1_neg'],
              datapoints['Ex2_pos'],
              datapoints['Ex2_neg'])
    #save_fig("DATA_PLOT_2")
    plt.show(block=False)

    # Features and target labels
    Xtrain = df.values[:,[0,1]]
    ytrain = df.values[:,[2]].reshape(-1,1)
    x1 = Xtrain[:,0]  # The first column of X
    x2 = Xtrain[:,1]  # The second column of X

    # Tranform X features to polynomial features
    Xmapped = mapFeature(x1,x2)
    init_theta = np.zeros((Xmapped.shape[1],1))

    # Creating an object of the Logistic Regression
    lr = LogisticRegression()
    input("Press <ENTER> to compute initial cost ...")
    cost = lr.cost_function(init_theta, Xmapped, ytrain,Lambda=1)
    print('\nInitial cost is: %f\n' % (cost))

    # Optimizing the cost function
    input("Press <ENTER> to train logistic regression\n")
    result = lr.fit(init_theta,Xmapped,ytrain,Lambda=1)
 
    # Evaluating model by computing the accuracy
    input("\nPress <ENTER> to compute training accuracy ...")
    y_hat = lr.predict(Xmapped,result.x)  
    correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0))\
               else 0 for (a, b) in zip(y_hat, ytrain)]  
    accuracy = (sum(map(int, correct)) % len(correct))  
    print('\nAccuracy = {0}%'.format(accuracy))

    # Plot the decision boundary
    input("\nPress <ENTER> to plot the decision boundary ...")
    plot_decision_boundary(result.x,1,datapoints)
    #save_fig("DECISION_BOUNDARY_2")
    plt.show(block=False)

    # Terminating program
    input("Press <ENTER> to terminate program ...")

if __name__ == "__main__":
    main()

    
