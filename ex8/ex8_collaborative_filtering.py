# Python ≥3.5 is required
import sys
assert sys.version_info >= (3, 5)

# Common imports
import scipy.io as sio
import numpy as np
import scipy.optimize as opt

# To suppress warnings
import warnings
warnings.filterwarnings('ignore')

def cofiCostFunc(params, Y, R, num_users, num_movies, num_features, Lambda):
    # Unfold the X and Theta matrices from params
    X = np.reshape(params[0:num_movies*num_features], (num_movies, num_features));
    Theta = np.reshape(params[num_movies*num_features:], (num_users, num_features));
    # Computing the cost
    pred = X.dot(Theta.T);
    error = pred - Y;
    error_factor = np.multiply(error,R);
    sqr_error = error_factor**2;
    total = np.sum(sqr_error);
    reg_term1 = (Lambda/2.) * np.sum(np.square(X))
    reg_term2 = (Lambda/2.) * np.sum(np.square(Theta))
    cost = (1/2) * total;                                        
    return cost + reg_term1 + reg_term2

def cofiGradFunc(params, Y, R, num_users, num_movies, num_features, Lambda):
    # Unfold the X and W matrices from params
    X = np.reshape(params[0:num_movies*num_features], (num_movies, num_features));
    Theta = np.reshape(params[num_movies*num_features:], (num_users, num_features));

    # Computing the gradient
    pred = X.dot(Theta.T);
    error = pred - Y;
    error_factor = np.multiply(error,R);
    sqr_error = error_factor**2;
    reg_x_grad = X * Lambda
    reg_theta_grad = Theta * Lambda
    X_grad = error_factor.dot(Theta) + reg_x_grad;              #regularised X gradients
    Theta_grad = error_factor.T.dot(X) + reg_theta_grad;        #regularised Theta gradients
    grads = np.hstack((X_grad.flatten(), Theta_grad.flatten())) # Regularized gradients
    return grads

def loadMovieList():
    movie_list =[]
    with open("./data/movie_ids.txt") as f:
        for line in f:
            movie_list.append(line[line.index(' ') + 1:].rstrip())
    return movie_list

# Function to normalize ratings
def normalizeRatings(Y, R):
    (m, n) = Y.shape;
    Ymean = np.zeros((m, 1));
    Ynorm = np.zeros(Y.shape);
    for i in range(m):
        idx = np.nonzero(R[i, :] == 1);
        Ymean[i] = Y[i, idx].mean(axis=1);
        Ynorm[i, idx] = Y[i, idx] - Ymean[i];
    return Ynorm, Ymean

def train_collaborative_filtering(Ynorm, R, num_users, num_movies, num_features, Lambda):
    # Set Initial Parameters (Theta, X)
    X     = np.random.randn(num_movies, num_features);
    Theta = np.random.randn(num_users, num_features);
    initial_parameters = np.hstack((X.flatten(), Theta.flatten()));
    
    # create cost and grad functions
    cost = lambda p: cofiCostFunc(p, Ynorm, R, num_users, num_movies,num_features, Lambda);
    grad = lambda p: cofiGradFunc(p, Ynorm, R, num_users, num_movies,num_features, Lambda).flatten();
    
    return opt.fmin_cg(cost, initial_parameters.T, fprime=grad, maxiter=500, disp=True)

def main():
    print("Programming Exercise 8: Recommender systems with Collaborative Filtering")
    # Load movie dataset
    data    = sio.loadmat('data/ex8_movies.mat' )
    Y = data['Y']
    R = data['R']

    print('\ni. Y is a %d x %d matrix which stores the ratings(from 1 to 5) of %d movies on %d users.\n'\
      %(Y.shape[0],Y.shape[1],Y.shape[0],Y.shape[1]))
    print('ii. R is a %d x %d binary-valued indicator matrix, where R(i,j) = 1\
    if user j gave a rating to movie i,\n and R(i,j) = 0 otherwise\n' %(R.shape[0],R.shape[1]))
    print('\nAverage rating for movie 1 (Toy Story): %f' % (Y[0,:].mean(axis=0)))

    # Load pre-trained weights (X, Theta, num_users, num_movies, num_features)
    param_path = 'data/ex8_movieParams.mat' 
    param_data    = sio.loadmat(param_path)
    X     = param_data['X']
    Theta = param_data['Theta']

    num_users = 4; num_movies = 5; num_features = 3;
    X     = X[0:num_movies, 0:num_features];
    Theta = Theta[0:num_users, 0:num_features];
    Y     = Y[0:num_movies, 0:num_users];
    R     = R[0:num_movies, 0:num_users];
    params = np.hstack((X.flatten(),Theta.flatten()))

    #  Evaluate cost function
    unregularized_cost     = cofiCostFunc(params, Y, R, num_users, num_movies,num_features, 0.0);
    unregularised_grads = cofiGradFunc(params, Y, R, num_users, num_movies,num_features, 0.0);
    regularized_cost       = cofiCostFunc(params, Y, R, num_users, num_movies,num_features, 1.5);
    regularized_grads   = cofiGradFunc(params, Y, R, num_users, num_movies,num_features, 1.5);
    print('\nCost at loaded parameters(lambda = 0.0): %f \
             \n(this value should be about 22.22)\n'%(unregularized_cost));
    print('Cost at loaded parameters(lambda = 1.5): %f \
             \n(this value should be about 31.34)\n'% (regularized_cost));

    movie_list = loadMovieList();
    # Initialize my ratings
    my_ratings = np.zeros((len(movie_list), 1));

    # Check the file movie_idx.txt for id of each movie in our dataset
    # For example, Toy Story (1995) has ID 1, so to rate it "4", you can set
    my_ratings[1] = 4;
    my_ratings[98] = 2;
    my_ratings[7] = 3;
    my_ratings[12]= 5;
    my_ratings[54] = 4;
    my_ratings[64]= 5;
    my_ratings[66]= 3;
    my_ratings[69] = 5;
    my_ratings[183] = 4;
    my_ratings[226] = 5;
    my_ratings[355]= 5;

    print('\nNEW USER RATINGS:\n');
    for i in range(len(my_ratings)):
        if my_ratings[i] > 0: 
            print('\tRated %d for %s\n' % (my_ratings[i],movie_list[i]))

    #  Import and load movie data
    #  Y is a 1682x943 matrix, containing ratings (1-5) of 1682 movies by 943 users
    #  R is a 1682x943 matrix, where R(i,j) = 1 if and only if user j gave a rating to movie i
    movie_data    = sio.loadmat('data/ex8_movies.mat')
    Y = movie_data['Y']
    R = movie_data['R']

    Y = np.hstack((my_ratings.reshape(len(movie_list), 1), Y))
    R = np.hstack((my_ratings.reshape(len(movie_list), 1) != 0, R))

    Ynorm, Ymean = normalizeRatings(Y, R)
    
    #  Useful Values
    num_users = Y.shape[1];
    num_movies = Y.shape[0];
    num_features = 10;

    # Set Initial Parameters (Theta, X)
    X     = np.random.randn(num_movies, num_features);
    Theta = np.random.randn(num_users, num_features);
    initial_parameters = np.hstack((X.flatten(), Theta.flatten()));

    # Fit collaborative filtering
    Lambda = 10
    result = train_collaborative_filtering(Ynorm, R, num_users, num_movies, num_features, Lambda)

    X     = result[0:num_movies * num_features].reshape((num_movies, num_features))
    Theta = result[num_movies * num_features:].reshape((num_users, num_features))

    p = X.dot(Theta.T)
    my_predictions = p[:,0]+ Ymean.ravel()
    idx = my_predictions.argsort()[::-1]
    print('\nTOP RECOMMENDATION FOR YOU:')
    for i in range(10):
        j = idx[i]
        print('Predicting rating %.1s for movie %s'% (my_predictions[j], movie_list[j]))


    print('\nOriginal ratings provided:')
    (index, _) = np.nonzero(my_ratings)
    for i in range(len(index)):
        j = index[i]
        print('Rated %.1s for movie %s'% (my_ratings[j,0], movie_list[j]))
    
    # Terminate program
    input("Press <ENTER> to terminate program ...")
    

if __name__ == "__main__":
    main()
