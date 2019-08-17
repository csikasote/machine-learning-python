# Python ≥3.5 is required
import sys
assert sys.version_info >= (3, 5)

# Common imports
import numpy as np
import scipy.io as sio
import os

# For data plots and visualizations
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

class NeuralNetwork(object):

    def sigmoid(self,z):
        return (1/(1+ np.exp(-z)))

    def sigmoid_gradient(self,z):
        return np.multiply(self.sigmoid(z),(1 - self.sigmoid(z)))

    def _one_hot(self,y,num_labels):
        y_matrix = np.zeros((len(y), num_labels))
        for i in range(len(y)):
            y_matrix[i, y[i] - 1] = 1
        return y_matrix
    
    def random_init_weights(self,L_in, L_out):
        epsilon_init = np.sqrt(6)/np.sqrt(L_in + L_out);
        return np.random.randn(L_out, 1 + L_in) * 2 * epsilon_init - epsilon_init;

    def initialize_parameters(self, nn_weights, input_layer_size, hidden_layer_size, num_labels):
        # -------------------------------------------------------------
        # Reshaping W1 and W2
        # -------------------------------------------------------------
        W1 = np.reshape(nn_weights[0:hidden_layer_size * (input_layer_size + 1)],\
                        (hidden_layer_size, (input_layer_size + 1)));
        W2 = np.reshape(nn_weights[(hidden_layer_size * (input_layer_size + 1)):,],\
                        (num_labels, (hidden_layer_size + 1)));
        
        assert (W1.shape == (hidden_layer_size, input_layer_size + 1))
        assert (W2.shape == (num_labels, hidden_layer_size + 1))

        parameters = {"W1":W1,
                      "W2":W2}
        
        return parameters
        
    def feedForward(self,X, parameters):
        
        # -------------------------------------------------------------
        # Retrieving parameters
        # -------------------------------------------------------------
        W1 = parameters["W1"]
        W2 = parameters["W2"]
            
        # -------------------------------------------------------------
        # The Forward Propagation Implementation
        # -------------------------------------------------------------
        A1 = np.insert(X,0,1,axis=1);
        Z2 = np.dot(A1,W1.T); 
        A2 = np.insert(self.sigmoid(Z2),0,1,axis=1);
        Z3 = np.dot(A2,W2.T);
        A3 = self.sigmoid(Z3);
        
        # -------------------------------------------------------------
        # SAVING THE ACTIVATIONS TO DICTIONARY
        # -------------------------------------------------------------
        cache = {"A1":A1,
                 "Z2":Z2, 
                 "A2":A2, 
                 "Z3":Z3,
                 "A3":A3}

        return A3, cache
                
    def nnCostFunction(self,A3, y, parameters,Lambda):
        
        # -------------------------------------------------------------
        # Retrieving parameters
        # -------------------------------------------------------------
        W1 = parameters["W1"]
        W2 = parameters["W2"]
        # -------------------------------------------------------------
        # Mapping vector y into a binary vector of 1's and 0's 
        # -------------------------------------------------------------
        y_matrix = np.zeros((len(y), 10))
        for i in range(len(y)):
            y_matrix[i, y[i] - 1] = 1
        
        # -------------------------------------------------------------
        # The Cost Implementation
        # -------------------------------------------------------------
        reg_term = (Lambda/(2*len(y))) * (np.sum(np.sum(np.square(W1[:,1:])))\
                                     + np.sum(np.sum(np.square(W2[:,1:])))); 
        cost = ((1/len(y) * np.sum(np.sum((np.multiply(-y_matrix,np.log(A3))\
                                      - np.multiply((1-y_matrix),np.log(1-A3))))))\
                + reg_term);
        
        return cost

    def backward_propagation(self,parameters, cache, X, y, Lambda, learning_rate):
        # -------------------------------------------------------------
        # Retrieving parameters
        # -------------------------------------------------------------
        W1 = parameters["W1"]
        W2 = parameters["W2"]

        # -------------------------------------------------------------
        # Retrieving cache parameters
        # -------------------------------------------------------------
        A1 = cache["A1"]
        Z2 = cache["Z2"]
        A2 = cache["A2"]
        Z3 = cache["Z3"]
        A3 = cache["A3"]
        
        # -------------------------------------------------------------
        # Mapping vector y into a binary vector of 1's and 0's 
        # -------------------------------------------------------------
        y_matrix = np.zeros((len(y), 10))
        for i in range(len(y)):
            y_matrix[i, y[i] - 1] = 1

        # -------------------------------------------------------------
        # Backpropagation algorithm
        # -------------------------------------------------------------
        d3 = A3-y_matrix;                               
        u = self.sigmoid(Z2);                            
        sig_grad = np.multiply(u,(1-u))                 
        d2 = np.multiply(np.dot(d3, W2[:,1:]),sig_grad) 
        delta1 = np.dot(d2.T,A1)                        
        delta2 = np.dot(d3.T,A2)
        temp1=W1; 
        temp2=W2;
        temp1[:,0]=0;
        temp2[:,0]=0;
        dW1 = (1/len(y) * delta1) + ((Lambda/len(y)) * temp1)
        dW2 = (1/len(y) * delta2) + ((Lambda/len(y)) * temp2)
        
        # -------------------------------------------------------------
        # Update parameters
        # -------------------------------------------------------------
        W1 = W1 - learning_rate * dW1
        W2 = W2 - learning_rate * dW2
        
        
        parameters = {"W1":W1,
                      "W2":W2}
        
        return parameters
    
    def model(self, X,y,initial_nn_params, input_layer_size, hidden_layer_size, num_labels, Lambda, learning_rate, print_cost=False):

        parameters = self.initialize_parameters(initial_nn_params,
                                           input_layer_size,
                                           hidden_layer_size,
                                           num_labels)
        
        W1 = parameters["W1"]
        W2 = parameters["W2"]
  
        cost_vec = [] # To store cost values per iterations
        print('\nNeural network learning in progress ...\n')
        print('Iteration\t\tCost\n==========\t\t====')
        for i in range((len(y))):
            A3, cache = self.feedForward(X,parameters)
            cost = self.nnCostFunction(A3, y, parameters, Lambda)
            cost_vec.append(cost)
            parameters = self.backward_propagation(parameters,cache, X, y, Lambda, learning_rate)
            if print_cost and i % 1000 ==0:
                print("%i\t\t\t%f"%(i,cost))      
        return parameters, np.array(cost_vec)

    def predict(self,X,parameters):
        A3, cache = self.feedForward(X,parameters)
        return np.argmax(A3, axis=1) + 1

# Visualize sample digits
def visualize_data(x):
    fig, ax = plt.subplots(nrows =5, ncols=5,sharex=True, sharey=True)
    h = plt.figure(1)
    ax = ax.flatten()
    m = x.shape[0]
    for i in range(25):
        img = x[np.random.randint(0,m),:].reshape(20,20,order="F")
        ax[i].imshow(img, cmap=mpl.cm.binary, interpolation="nearest")
        ax[i].axis("on")
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()
    plt.show(block=False)

def nnLearningCurve(cost_vec):
    g = plt.figure(2)
    plt.plot(range(len(cost_vec)),cost_vec,'b-o', label= r'${J{(\theta)}}$')
    plt.grid(True)
    plt.title("Neural Network cost convergence graph")
    plt.xlabel('# of Iterations')
    plt.ylabel(r'${J{(\theta)}}$', rotation=1)
    plt.xlim([-1000,len(cost_vec)])
    plt.ylim([0,7])
    plt.legend()

def main():

    # Instantiating object of the Neural Network class
    nn = NeuralNetwork()

    # Testing sigmoid gradient
    input('Press <ENTER> to evaluate sigmoid gradient function ...')
    z = np.array([[-1, -0.5, 0, 0.5, 1]]);
    print('\nFor z:',z)
    print('\nGradients for z:\n',str(nn.sigmoid_gradient(z)));
    print('\nGradient at [z = 0]:',str(nn.sigmoid_gradient(z)[0,2]));

    # Import dataset into variables X
    input("\nPress <ENTER> to load dataset into X and y variables ...")
    data = sio.loadmat('data/ex4data1.mat')
    X = data['X']
    y = data['y']
    print('\nLoaded data successfully ...')
    print('\nX(%d, %d)'\
          % (X.shape[0], X.shape[1]), end='')
    print(', y(%d, %d)'\
          % (y.shape[0], y.shape[1]))

    # Visualize dataset
    input("\nPress <ENTER> to visualize sample digits ...")
    visualize_data(X)
    save_fig("SAMPLE_DIGITS")

    # Load the weights into variables W1 and W2
    input("\nPress <ENTER> to load NN parameters ...")
    weights = sio.loadmat('data/ex4weights.mat')
    W1 = weights['Theta1']
    W2 = weights['Theta2']
    W1_flat = W1.flatten()
    W2_flat = W2.flatten()

    # Unroll the loaded parameters
    nn_weights = np.concatenate((W1_flat,W2_flat),axis=0).reshape(-1,1)
    print('\nNeural Network Parameters Successfully Loaded ...\n')

    # Neural Network model structure
    input_layer_size  = 400
    hidden_layer_size = 25
    num_labels = 10

    # Retrieve network parameters 
    parameters = nn.initialize_parameters(nn_weights,
                                       input_layer_size,
                                       hidden_layer_size,
                                       num_labels);

    W1 = parameters["W1"]
    W2 = parameters["W2"]
    print('W1' + str(W1.shape), end='')
    print(', W2' + str(W2.shape))

    # Unregulerised cost
    input("\nPress <ENTER> to compute unregularised cost [Lambda=0] ...")
    Lambda = 0 #No regularisation
    A3, cache = nn.feedForward(X,parameters)
    cost = nn.nnCostFunction(A3, y, parameters, Lambda)
    print('\nCost(with loaded parameters from ex4weights): %.6f' % (cost))

    # Regularised cost
    input("\nPress <ENTER> to compute regularised cost [Lambda=1] ...")
    Lambda = 1 #With regularisation
    A3, cache = nn.feedForward(X,parameters)
    cost = nn.nnCostFunction(A3, y, parameters, Lambda)
    print('\nRegularised cost(with loaded parameters from ex4weights): %.6f' % (cost))


    # Initializing NN Weights for training
    input("\nPress <ENTER> to initialized NN parameters ...")
    print('\nInitializing Neural Network Parameters ...\n')
    W1 = nn.random_init_weights(input_layer_size, hidden_layer_size);
    W2 = nn.random_init_weights(hidden_layer_size, num_labels);
    nn_params = np.concatenate((W1.flatten(),W2.flatten()),axis=0).reshape(-1,1)
    print('W1'+str(W1.shape), end='')
    print(', W2'+str(W2.shape), end='')
    print(', nn_params'+str(nn_params.shape))

    # Training Neural Network
    input("\nPress <ENTER> to train the Neural Network ...")
    parameters, cost_vec = nn.model(X,y,
                                    nn_params,
                                    input_layer_size,
                                    hidden_layer_size,
                                    num_labels,
                                    Lambda=1,
                                    learning_rate=1,
                                    print_cost=True)
    # Plot learning curve
    input("\nPress <ENTER> to plot learning curve ...")
    nnLearningCurve(cost_vec)
    save_fig("LEARNING_CURVE")
    plt.show(block=False)
    
    # Training accuracy
    input("\nPress <ENTER> to compute NN training accuracy ...")
    y_hat = nn.predict(X, parameters)
    correct = [1 if a == b else 0 for (a, b) in zip(y_hat, y)]
    accuracy = (sum(map(int, correct)) / float(len(correct)))
    print('\nAccuracy (obtained by NN): %.2f' % (accuracy * 100)+'%\n')


    # Terminate program
    input("\nPress <ENTER> to terminate program ...")

    

if __name__ == "__main__":
    main()
