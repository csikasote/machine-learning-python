# Import common python libraries
from scipy import optimize as opt
import pandas as pd
import numpy as np
import os

# Matplotlib for plotting figures
import matplotlib as mpl
import matplotlib.pyplot as plt


# Where to save the figures
EXERCISE_ROOT_DIR = "."
IMAGES_PATH = os.path.join(EXERCISE_ROOT_DIR, "images")

# Defining a class Logistic Regression that implements ex2
class LogisticRegression(object):
    def __init__(self, n_iter=500):
        self.n_iter   = n_iter

    # Sigmoid function
    def sigmoid(self, z):
        return (1/(1+ np.exp(-z)))

    # Net Input
    def net_input(self,X,theta):
        return np.dot(X,theta)
    
    # Objective function
    def cost_function(self,theta,X,y):
        z = self.net_input(X,theta)
        y_hat = self.sigmoid(z)
        return (-1/len(X)) * (np.dot(y.T,np.log(y_hat)) +\
                              np.dot((1-y).T,np.log(1-y_hat)))

    # Using "fmin()" to optimize
    def fit(self,X,y,theta):
        return opt.fmin(func=self.cost_function,
                        x0=theta,
                        args=(X,y),
                        maxiter=self.n_iter,
                        full_output=True)
    # Predict function
    def predict(self,X,theta):
        z = self.net_input(X,theta)
        probability = self.sigmoid(z)
        return [1 if p >= 0.5 else 0 for p in probability]

    # Classification function
    def classify(self,X,theta):
        z = self.net_input(X,theta)
        probability = self.sigmoid(z)
        return ["ADMIT" if p >= 0.5 else "DECLINE" for p in probability]

    # Decision function
    def decision_(self,X,theta):
        z = self.net_input(X,theta)
        return "\nProbability: %2f"%(self.sigmoid(z)[0])\
               + ' \nPrediciton: ' + str(self.predict(X,theta)[0]) \
               + ' \nModel Decision: ' + str(self.classify(X,theta)[0]) \

# Function to plot the data distribution
def plot_data(x1,x2,y1,y2): 
    plt.scatter(x1,y1, s=50, c='b', marker='o', label='Admitted')  
    plt.scatter(x2,y2, s=50, c='r', marker='x', label='Not Admitted')  
    plt.xlabel('Exam 1 Score')  
    plt.ylabel('Exam 2 Score')
    plt.title('Scatter plot of Trainig data')
    plt.axis([20,110,20,110])
    plt.grid(True)
    plt.legend()

# Plot decision boundary function
def plot_decision_boundary(x1,x2,y1,y2,theta):
    theta0 = theta[0]; 
    theta1 = theta[1]; 
    theta2 = theta[2];
    fig, ax = plt.subplots()
    plt.scatter(x1,y1, s=50, c='b', marker='o', label='Admitted')  
    plt.scatter(x2,y2, s=50, c='r', marker='x', label='Not Admitted')
    x_vals = np.array(ax.get_xlim())
    y_vals = -1 * np.divide(((np.multiply(theta1,x_vals)) + theta0),theta2)
    plt.plot(x_vals, y_vals, '--', c="red", label='Decision Boundary')
    titlestr = 'Decision Boundary Function: y = %.2f + %.2fX1 + %.2fX2' % (theta0, theta1, theta2)
    plt.title(titlestr) 
    plt.axis([20,110,20,110])
    plt.xlabel('Exam 1 Score')  
    plt.ylabel('Exam 2 Score')
    plt.grid(True)
    plt.legend()

# The function allows images to be saved
def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

def main():
    # Load dataset
    path = os.getcwd() + '\data\ex2data1.txt'
    df = pd.read_csv(path, header=None, names=['Exam 1', 'Exam 2', 'Admitted'])
    pos_df = df[df['Admitted'].isin([1])]
    neg_df = df[df['Admitted'].isin([0])]
    Ex1_pos = pos_df['Exam 1'] 
    Ex2_pos = pos_df['Exam 2']
    Ex1_neg = neg_df['Exam 1']
    Ex2_neg = neg_df['Exam 2']

    # Visualize the data distribution
    input("Press <ENTER> to visualize data ...")
    plot_data(Ex1_pos,Ex1_neg,Ex2_pos,Ex2_neg)
    save_fig("DATA_PLOT")
    plt.show()

    # Extracting Features and target labels
    X = df.values[:,[0,1]] 
    y = df.values[:,[2]]
    
    # Add a column of ones to x for x0 = 1
    X = np.c_[np.ones((len(X), 1)), X]
    init_theta = np.zeros((X.shape[1],1))

    # Instantiate an object the LogisticRegression class
    lr = LogisticRegression()
    
    # Print the initial cost
    print('\nInitial Cost: %f\n' % (lr.cost_function(init_theta,X,y)))

    # Optimizing the cost function
    result = lr.fit(X,y,init_theta)
    print('\nThe minimum point found: ',result[0])
    print('\nComputed cost at minimum point: %.6f\n' %(result[1]))

    # Plotting decision boundary
    input("Press <ENTER> to plot decision boundary ...")
    plot_decision_boundary(Ex1_pos,Ex1_neg,Ex2_pos,Ex2_neg,result[0])
    #save_fig("DECISION_BOUNDARY")
    plt.show()

    # Testing model with predictions
    input("\nPress <ENTER> key to test the model ...")
    Xtest = [[1, 45, 85]] #TEST EXAMPLE
    decision = lr.decision_(Xtest,result[0])
    print(decision)
    input('\nPress <ENTER> to terminate program ...')

if __name__ == "__main__":
    main()

    
