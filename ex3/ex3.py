# Python ≥3.5 is required
import sys
assert sys.version_info >= (3, 5)

# Import common libraries
from scipy import optimize as opt
import scipy.io as sio
import numpy as np
import os

# For visualization
import matplotlib
import matplotlib.pyplot as plt

# Where to save the figures
EXERCISE_ROOT_DIR = "."
IMAGES_PATH = os.path.join(EXERCISE_ROOT_DIR, "images")

# PART 1: MULTICLASS CLASSIFICATION
class LogisticRegression(object):

    def __init__(self, n_iters=500):
        self.n_iters = n_iters

    # Sigmoid function
    def sigmoid(self, z):
        return (1/(1+ np.exp(-z)))

    # Net Input function
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

    # Gradient function
    def gradient_function(self, theta, X, y, Lambda):
        y_hat = self.sigmoid(self.net_input(X, theta));
        grad = (1/len(y)) * np.dot(X.T,(y_hat-y));
        temp=theta;
        temp[0]=0; #forcing theta(1) to be zero
        reg_term= temp * Lambda/len(y)
        return grad + reg_term

    # Optimization with minimize() function
    def oneVsAll(self,X, y, num_labels, Lambda):
        x = np.insert(X,0,1,axis=1)
        n = X.shape[1]
        all_theta = np.zeros((num_labels, n));
        print('\nInitializing optimization process ...\n')
        for c in range(num_labels):
            initial_theta= np.zeros((n, 1));
            z = 10 if c == 0 else c
            logic_y = np.array([1 if u == z else 0 for u in y])
            print(str(c+1) + '. Optimizing for handwritten number %d'% z +' ... DONE')
            args=(X,logic_y,Lambda)
            x0 = initial_theta
            opts = {'maxiter' : None, 
                    'disp' : False, 
                    'gtol' : 1e-5, 
                    'norm' : np.inf,
                    'eps' : 1.4901161193847656e-08} 
            result = opt.minimize(self.cost_function, x0, jac=self.gradient_function, args=args, 
                                  method='CG', options=opts)
            all_theta[[c],:]= (result.x).T;
        print('\nOPTIMIZATION PROCESS COMPLETED SUCCESSFULLY!!!')
        return all_theta

    # Prediction function
    def predict(self,all_theta, X):
        m = X.shape[0];
        num_labels = all_theta.shape[0];
        prediction = np.zeros((m, 1));
        h = self.sigmoid(self.net_input(X, all_theta.T));
        prediction = np.argmax(h, axis=1)
        prediction[prediction == 0] = 10
        return prediction


# Display sample images of digits
def display_data(sel):
    fig, ax = plt.subplots(nrows =5, ncols=5, sharex=True, sharey=True,)
    ax = ax.flatten()
    for i in range(25):
        img = sel[i].reshape(20,20,order="F")
        ax[i].imshow(img, cmap=matplotlib.cm.binary, interpolation="nearest")
        ax[i].axis('on')
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()

# The function allows images to be saved
def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

# PART 2: NEURAL NETWORKS
class NNPredict(object):

    # Sigmoid function
    def sigmoid(self, z):
        return (1/(1+ np.exp(-z)))
    # Neural network prediction function //FEEDFORWARD
    def predictNN(self,X,Theta1, Theta2):
        # Feedforward
        a1 = np.insert(X,0,1,axis=1)
        z2 = np.dot(a1, Theta1.T);
        a2 = np.insert(self.sigmoid(z2),0,1,axis=1)
        z3 = np.dot(a2,Theta2.T);
        a3 = self.sigmoid(z3);
        return np.argmax(a3, axis=1) + 1


def main():

    # PART 1: Multiclass classification
    print("PART 1: Multiclass Classification (using Logistic Regression)")
    # Load dataset
    data = sio.loadmat('data/ex3data1.mat')
    X = data['X']
    y = data['y']

    # Randomly select images
    rand_indices = np.random.permutation(X.shape[0]);
    sel = X[rand_indices[:25], :]

    # Visualize digits
    input("\nPress <ENTER> to visualize digits ...")
    display_data(sel)
    #save_fig("DIGITS_IMG")
    plt.show(block=False)

    # Initial test settings
    Testtheta = np.array([[-2],[-1],[1],[2]])
    Xtest = np.reshape(np.array(range(1,16)), (3,5)).T/10 
    Xtest = np.insert(Xtest,0,1,axis=1)
    ytest = (np.array([[1],[0],[1],[0],[1]])>= 0.5)

    
    # Instantiating an object the class Logistic regression
    lr = LogisticRegression()
    
    # Compute initial cost and gradient with initial test settings
    input("\nPress <ENTER> to compute initial cost ...")
    cost = lr.cost_function(Testtheta, Xtest, ytest, Lambda=3)
    print('\nComputed cost is: %.6f' % (cost))

    input("\nPress <ENTER> to compute initial gradient ...")
    grad = lr.gradient_function(Testtheta, Xtest, ytest, Lambda=3)
    print('\nComputed gradient is:\n\n' + str(grad)+'\n')


    # Training parameters
    Lambda= 0.1 #Lambda hyperparameter
    num_labels = 10 # Number of labels

    # Running oneVsAll function
    input("\nPress <ENTER> to run one-vs-all process ...")
    all_theta = lr.oneVsAll(X, y, num_labels, Lambda)

    # Computing the training accuracy
    input("\nPress <ENTER> to compute training accuracy ...")
    y_hat = lr.predict(all_theta, X)
    correct = [1 if a == b else 0 for (a, b) in zip(y_hat, y)]
    accuracy = (sum(map(int, correct)) / float(len(correct)))
    print('\nAccuracy computed (using Logistic regression) is: %.2f'\
          % (accuracy * 100)+'%')
    # Continue to part two of the programming exercise
    input("\nPress <ENTER> to continue ...\n")
    
    # PART 2: NEURAL NETWORK
    print("\nPART 2: Multiclass Classification (using ANN)")
    # Load weights for the neural network
    input("\nPress <ENTER> to load weights ...")
    param = sio.loadmat('data/ex3weights.mat')
    theta1 = param['Theta1']
    theta2 = param['Theta2']
    print('\nNeural Network Parameters Successfully Loaded ...')

    # NEURAL NETWORK PREDICTION
    input("\nPress <ENTER> to compute NN accuracy ...")
    y_hat = NNPredict().predictNN(X, theta1, theta2)
    correct = [1 if a == b else 0 for (a, b) in zip(y_hat, y)]
    accuracy = (sum(map(int, correct)) / float(len(correct)))
    print('\nAccuracy computed by NN (using loaded weights) is: %.2f' % (accuracy * 100)+'%')

    # Terminate the program
    input("\nPress <ENTER> to terminate program")

if __name__ == "__main__":
    main()
    

    
        
    
