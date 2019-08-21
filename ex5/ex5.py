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
        reg_term = (self.Lambda/(2*len(y))) *\
                   np.sum(np.dot(theta[1:].T,theta[1:]))
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
        theta = opt.fmin_cg(cost, initial_theta.T, fprime=grad, maxiter=150, disp=False)
        return theta.reshape(-1,1)
    
    def learningCurve(self,X, y, Xval, yval,theta):
        error_train = np.zeros((len(y), 1))
        error_val   = np.zeros((len(y), 1))
        for i, j in enumerate(range(1,len(y)+1)):
            theta = self.trainLinearReg(X[0:j,:], y[:j,:]);
            error_train[i] = self.linearRegCostFunction(X[:j,:], y[:j,:], theta);
            error_val[i] = self.linearRegCostFunction(Xval, yval, theta);
        print('\n# Training Examples\tTrain Error\tCV Error\n');
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

# Map X features to Polynomial features
def polyFeatures(X, p):
    Xpoly = np.zeros((X.shape[0], p))
    Xpoly[:,0] = X[:,0]
    for i in range(1,p):
        Xpoly[:,i] = np.power(X[:,0],i+1); 
    return Xpoly

# Feature normalization
def featureNormalize(X):
    mu = np.mean(X, axis=0, keepdims=True);
    sigma = np.std(X,ddof=1, axis = 0, keepdims=True);
    Xnorm = (X - mu)/sigma
    return Xnorm, mu, sigma

# Plot polynomial line
def plotPolynomialGraph(min_x, max_x, mu, sigma, theta, p):
    x = np.arange(min_x-15, max_x+25, 0.05)
    x = np.reshape(x, (len(x), 1))

    x_poly_train = polyFeatures(x, p);
    x_poly_train = (x_poly_train - mu)/sigma;
    x_poly_train = np.insert(x_poly_train,0,1,axis=1)
    h = np.dot(x_poly_train, theta)
    plt.plot(x, h, 'b--')

# Fits the polynomial line on the training examples
def plotPolynomialFit(X,y, mu,sigma,theta,p,fig_num):
    i = plt.figure(fig_num)
    plt.plot(X,y,'rx',markersize=10, label='Training Example')
    plotPolynomialGraph(X.min(), X.max(), mu, sigma, theta,p)
    plt.xlabel('Change in water level (x)');
    plt.ylabel('Water flowing out of the dam (y)');
    titlestr = 'Polynomial Regression Fit (%s = %d)' % (r'$\lambda$',0)
    plt.title(titlestr)
    plt.grid(True)
    plt.axis([-100,100,-100,100])
    plt.legend()
    plt.show(block=False)

# Plot polynomial learning curves
def plotPolyLearningCurves(error_train, error_val,fig_num):
    j = plt.figure(fig_num)
    plt.plot(range(1, len(error_train) + 1), error_train, 'b-o', label='Train')
    plt.plot(range(1, len(error_train) + 1), error_val,'r-o', label='Cross Validation')
    plt.xlabel('Number of training examples')
    plt.ylabel('Error')
    plt.title('Polynomial Regression Learning Curve (%s = %d)' %\
              (r'$\lambda$',0));
    plt.grid(True)
    plt.legend()
    plt.axis([0,14,-20,100])
    plt.show(block=False)


def main():
    print("\nProgramming Exercise 5: Regularized Linear Regression and Bias vs.Variance\n")
    # Load training dataset for this exercise
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

    # Mapping X, Xval, Xtest onto Polynomial features
    X = data['X']
    poly_degree = 8

    # Map X onto polynomial features
    input("\nPress <ENTER> to map X, Xval, Xtest to polynomial features ...")
    X_poly_train = polyFeatures(X, poly_degree)
    X_poly_train, mu, sigma = featureNormalize(X_poly_train)
    X_poly_train = np.insert(X_poly_train,0,1,axis=1) #Add one the X_poly_train
    
    # Map Xval onto polynomial features
    Xval = data['Xval']
    X_poly_val = polyFeatures(Xval, poly_degree);
    X_poly_val = (X_poly_val - mu)/sigma
    X_poly_val = np.insert(X_poly_val,0,1,axis=1) #Add one the X_poly_val
    
    # Map Xtest onto polynomial features
    X_poly_test = polyFeatures(Xtest, poly_degree);
    X_poly_test = (X_poly_test - mu)/sigma
    X_poly_test = np.insert(X_poly_test,0,1,axis=1) #Add one the X_poly_test

    # Fit linear regression on polynomial features
    input("\nPress <ENTER> to fit linear regression on polynomial features ...")
    theta = lr.trainLinearReg(X_poly_train, y);
    error_train, error_val = lr.learningCurve(X_poly_train, y, X_poly_val, yval,theta);

    # Plot linear regression with polynomial features
    input("\nPress <ENTER> to plot polynomial fit line ...")
    plotPolynomialFit(X,y, mu,sigma,theta,poly_degree,4)
    #save_fig("POLYNOMIAL_FIT")

    # Plot the polynomial learning curves
    input("\nPress <ENTER> to plot polynomial learning curves ...")
    plotPolyLearningCurves(error_train, error_val,5)
    #save_fig("POLYNOMIAL_LEARNING_CURVE")   

    # Terminate program
    input("\nPress <ENTER> to terminate program ...")


if __name__ == "__main__":
    main()
