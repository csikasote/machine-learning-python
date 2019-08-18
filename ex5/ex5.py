# Python ≥3.5 is required
import sys
assert sys.version_info >= (3, 5)

# Common imports
import scipy.io as sio
import numpy as np
import os

# For optimization
from scipy import optimize as opt

# For data plots and visualization
import matplotlib.pyplot as plt
import matplotlib as mpl

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

class LinearRegression(object):
    def __init__(self,Lambda):
        self.Lambda = Lambda
        
    def hypothesis(self,X,theta):
        return np.dot(X,theta)

    def linearRegCostFunction(self,X, y, theta):
        y_hat = self.hypothesis(X,theta)
        cost =(1/(2*len(y))) * np.sum(np.square(y_hat-y));
        temp = np.ones((theta.shape[0],1))
        temp[0,0] = 0
        reg_term = (self.Lambda/(2*len(y)))\
                   * np.sum(np.square(np.multiply(temp, theta)))
        return cost + reg_term
    
    def linearRegGradientFunction(self,X, y,theta):
        y_hat = self.hypothesis(X,theta)
        error = (y_hat - y)
        grad = (1/len(y)) * np.dot(X.T,error)
        temp = np.ones((theta.shape[0],1))
        temp[0,0] = 0.0
        reg_term = (self.Lambda/len(y))\
                   * np.multiply(temp,theta)
        return grad + reg_term

    def trainLinearReg(self,X, y):
        initial_theta = np.zeros((X.shape[1], 1))
        cost = lambda t: self.linearRegCostFunction(X, y, t.reshape(-1,1))
        grad = lambda t: self.linearRegGradientFunction(X, y, t.reshape(-1,1)).flatten()
        theta = opt.fmin_cg(cost, initial_theta.T, fprime=grad, maxiter=200, disp=False)
        return theta.reshape(-1,1)
    
    def learningCurve(self,X, y, Xval, yval,theta):
        error_train = np.zeros((len(y), 1))
        error_val   = np.zeros((len(y), 1))
        for i, j in enumerate(range(1,len(y)+1)):
            theta = self.trainLinearReg(X[:j,:], y[:j,:]);
            error_train[i] = self.linearRegCostFunction(X[:j,:], y[:j,:], theta);
            error_val[i] = self.linearRegCostFunction(Xval, yval, theta);
        print('\n# Training Examples\tTrain Error\tCross Validation Error\n');
        for i in range(0, len(y)):
            print('  \t%d\t\t%f\t%f\n' % (i+1, error_train[i], error_val[i]));
        return error_train, error_val
    


# To plot the training data
def plot_data(X,y):
    f = plt.figure(1)
    plt.plot(X,y,'rx',markersize=10, label='Training Example')
    plt.grid(True) #Always plot.grid
    plt.ylabel('Water flowing out of the dam (y)')
    plt.xlabel('Change in water levels (x)')
    plt.title('Figure 1: Training Examples')
    plt.axis([-50, 40, -5, 40])
    plt.legend()
    plt.show(block=False)

# To plot the linear fit
def plotLinearFit(x,y,h,theta):
    g = plt.figure(2)
    plt.plot(x,y,'rx',markersize=10, label='Training Example')
    plt.plot(x,h,'b-',label = 'Hypothesis: h(x) = %0.2f + %0.2fx'%(theta[0],theta[1]))
    plt.grid(True)
    plt.ylabel('Water flowing out of the dam (y)')
    plt.xlabel('Change in water levels (x)')
    titlestr = 'Hypothesis Function: Y = %.2f + %.2fX' % (theta[0], theta[1])
    plt.title(titlestr)
    plt.legend()
    plt.axis([-50, 40, -5, 40])
    plt.show(block=False)

def plotLearningCurve(error_train,error_val):
    h = plt.figure(3)
    plt.plot(range(1,len(error_train)+1),error_train,'b-o', label='Train')
    plt.plot(range(1,len(error_train)+1),error_val,'r-o', label='Cross Validation')
    plt.xlabel('Number of training examples')
    plt.ylabel('Error')
    plt.title('Linear Regression Learning Curve (%s = %d)' % (r'$\lambda$',0))
    plt.grid(True)
    plt.legend()
    plt.show(block=False)

def main():
    # Load dataset
    input("\nPress <ENTER> to load dataset ...")
    data = sio.loadmat('data/ex5data1.mat')
    X,y = data['X'],data['y'] # Loading training data
    Xtest,ytest = data['Xtest'], data['ytest'] # Loading test data
    Xval, yval  = data['Xval'],data['yval']    # Loading validation data
    print('\nTraining data loaded successfully ...')
    
    # Visualize training data
    input("\nPress <ENTER> to visualize training data ...")
    plot_data(X,y)
    #save_fig("TRAINING_DATA")

    # Test settings
    X = np.insert(X,0,1,axis=1) 
    Xval = np.insert(Xval,0,1,axis=1)
    theta_test =np.array([[1],[1]])
    print('\nX' + str(X.shape))

    # Instantiate the LinearRegression class
    lr = LinearRegression(Lambda=1)

    # Compute cost with test settings
    input("\nPress <ENTER> to test cost function ...")
    cost = lr.linearRegCostFunction(X, y, theta_test)
    print('\nComputed cost at theta [[1],[1]]: %6f '% (cost));

    # Compute gradient with test settings
    input("\nPress <ENTER> to test gradient function ...")
    grad = lr.linearRegGradientFunction(X, y, theta_test);
    print('\nGradient at theta [[1],[1]]:',str(grad[:,0]))

    # Train linear regression
    input("\nPress <ENTER> to train linear regression ...")
    lr = LinearRegression(Lambda=0)
    theta = lr.trainLinearReg(X, y);
    print('\nMinimum theta found: %s '% (str(theta[:,0])))

    # Plot linear fit
    input("\nPress <ENTER> to plot the linear fit ...")
    p = lr.hypothesis(X,theta)
    plotLinearFit(X[:,1],y,p,theta)
    #save_fig("LINEAR_FIT")

    # Learning curves
    input("\nPress <ENTER> to evaluate model for plotting learning curves ...")
    error_train, error_val = lr.learningCurve(X, y, Xval, yval,theta)
    input("\nPress <ENTER> to plot learning curves ...")
    plotLearningCurve(error_train,error_val)
    #save_fig("LEARNING_CURVE_LR")

    #Terminate program
    input("\nPress <ENTER> to terminate program ...")

    

if __name__ == "__main__":
    main()
