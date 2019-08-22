# Python ≥3.5 is required
import sys
assert sys.version_info >= (3, 5)

# Common imports
import numpy as np
import scipy.io as sio

# To plot pretty figures
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# To suppress warnings
import warnings
warnings.filterwarnings('ignore')

def plot_data_points(X, y=None):
    plt.scatter(X[:, 0], X[:, 1], alpha=0.6, label='Dataset points')
    plt.tight_layout()
    plt.xlabel("Latency (ms)", fontsize=14)
    plt.ylabel("Throughput (mb/s)", fontsize=14)
    plt.axis([0.0, 30.0, 0.0, 30.0])
    plt.legend()
    plt.grid(True)

def estimateGaussian(X):
    mu = np.mean(X, axis= 0)
    sigma_sqrd = np.var(X, axis= 0)
    return mu, sigma_sqrd

def multivariateGaussian(X, mu, sigma_sqrd):
    n = X.shape[1]
    if n > 1:                                         
        sigma_sqrd = np.diag(sigma_sqrd)
    
    sigma_sqrd_det = np.linalg.det(sigma_sqrd)         
    sigma_sqrd_inv = np.linalg.pinv(sigma_sqrd)        
    
    X_diff = X - mu
    eqn_part1 = 1/((2*np.pi)**(n/2.0) * sigma_sqrd_det**(0.5))
    eqn_part2 = np.exp(-0.5 * (X_diff.dot(sigma_sqrd_inv) * X_diff).sum(axis=1))
    return eqn_part1 * eqn_part2

def visualizeFit(X,mu, sigma_sqrd):
    linespace = np.arange(0, 35.5, 0.5)
    xx, yy = np.meshgrid(linespace, linespace)
    Z = multivariateGaussian(np.c_[xx.ravel(), yy.ravel()],mu,sigma_sqrd);
    Z = Z.reshape(xx.shape)
    if np.sum(np.isinf(Z)) == 0:
        plt.contourf(xx, yy, Z, norm=LogNorm(vmin=Z.min(), 
                                             vmax=Z.max()),
                     levels=10.0 ** np.arange(-20, 0, 3))
        plt.contour(xx, yy, Z,  norm=LogNorm(vmin=Z.min(), 
                                             vmax=Z.max()),
                    levels=10.0 ** np.arange(-20, 0, 3),linewidths=1, colors='k')
    plot_data_points(X)

def selectThreshold(yval, pval):
    bestEpsilon = 0.0;
    bestF1 = 0.0;
    F1 = 0.0;
    stepsize = (np.max(pval) - np.min(pval)) / 1000
    for epsilon in np.arange(min(pval), max(pval), stepsize):
        cvPredictions = (pval < epsilon).astype(int).reshape(-1,1) 
        fp = cvPredictions[np.nonzero(yval == 0)].sum()       
        tp = cvPredictions[np.nonzero(yval == 1)].sum()       
        fn = yval[np.nonzero(cvPredictions == 0)].sum()       
        
        precision = (tp /(tp + fp));                                # Precision
        recall = (tp/(tp+fn));                                      # Computes the recall
        F1= (2.0 * precision*recall) / (precision + recall);        # Computes the F1 score
        
        if F1 > bestF1:
            bestF1 = F1
            bestEpsilon = epsilon
    return bestEpsilon,bestF1

def plotOutliers(X,outliers,mu,sigma_sqrd):
    visualizeFit(X,mu, sigma_sqrd)
    radius = (X.max() - X) / (X.max() - X.min())
    plt.scatter(X[outliers, 0], X[outliers, 1],
                facecolors='none', 
                edgecolors='r', s=1000 * radius, label="Outlier points")
    legend = plt.legend(loc='upper right')
    legend.legendHandles[0]._sizes = [10]
    legend.legendHandles[1]._sizes = [20]
  
def main():
    print("\nProgramming Exercise 8: Anomaly Detection\n")

    # Load dataset
    data    = sio.loadmat('data/ex8data1.mat' )
    X = data['X']
    Xval = data['Xval']
    yval = data['yval']

    # Plot dataset
    plt.figure(1)
    plot_data_points(X, y=None)
    plt.show(block=False)

    # Estimate mu and sigma
    mu, sigma_sqrd = estimateGaussian(X);
    p = multivariateGaussian(X, mu, sigma_sqrd)

    # Plot contours
    plt.figure(2)
    visualizeFit(X,mu,sigma_sqrd)
    plt.title("The Gaussian distribution contours")
    plt.show(block=False)

    # Cross Validation
    pval = multivariateGaussian(Xval, mu, sigma_sqrd);
    epsilon, F1 = selectThreshold(yval, pval)
    print('\nBest epsilon found using cross-validation: ' + str(epsilon));
    print('Best F1 on Cross Validation Set: %f.' % (F1));
    # Find the outliers in the training set and plot the
    outliers = np.where(p < epsilon)
    print('With computed epsilon %f, the # of outliers found is %d.'%\
          (epsilon,(p < epsilon).sum()))

    plt.figure(3)
    plotOutliers(X,outliers,mu,sigma_sqrd)
    plt.title('Plot computed outlier points')
    plt.show(block=False)

    # PART 2: High dimensional dataset
    data    = sio.loadmat('data/ex8data2.mat')
    X_train = data['X']
    X_val = data['Xval']
    y_val = data['yval']

    mu,sigma = estimateGaussian(X_train); #  Apply the same steps to the larger dataset 
    pval     = multivariateGaussian(X_val, mu, sigma); #  Cross-validation set
    epsilon,F1 = selectThreshold(y_val, pval); #  Find the best threshold
    print('Best epsilon found using cross-validation:\n\t', epsilon)
    print('Best F1 on Cross Validation Set:\n\t', F1)
    print('# Outliers found:\n\t', (p < epsilon).sum())

    # Terminate program
    input("Press <ENTER> key to terminate program ...")
        
if __name__ == "__main__":
    main()
