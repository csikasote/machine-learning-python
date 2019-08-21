# Python ≥3.5 is required
import sys
assert sys.version_info >= (3, 5)

# Common imports
import scipy.io as sio
import numpy as np
import os

# To plot pretty figures
import matplotlib as mpl
import matplotlib.pyplot as plt

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


def plot_data_points(X, y=None,ax = None):
    ax = ax or plt.gca()
    ax.scatter(X[:, 0], X[:, 1], 
                c='white', marker='o', 
                edgecolor='b', s=50,label='dataset points')
    plt.grid(True)
    ax.set_aspect('equal')
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    plt.axis([2.0, 7.0, 3.0, 8.0])
    plt.legend()

def feature_normalize(X):
    X_mu = X.mean(axis=0)
    X_std = X.std(axis=0) 
    X_norm = (X - X_mu) / X_std
    return X_norm, X_mu, X_std

def compute_cov_matrix(X_norm):
    Sigma = np.cov(X_norm.T)
    return Sigma

def decompose_cov_matrix(Sigma): 
    U,S,V = np.linalg.svd(Sigma) 
    return U,S,V

def project_data(X_norm,U,k):
    W = U.T[:,:k]           
    Z = X_norm.dot(W)  
    return W,Z

def plot_eigen_vectors(X,U,ax = None):
    X_mu = X.mean(axis=0)
    u1 = U[:,0].reshape(-1,1)
    u2 = U[:,1].reshape(-1,1)
    ax = ax or plt.gca()
    #plot_data_points(X)
    ax.scatter(X[:, 0], X[:, 1], 
                c='white', marker='o', 
                edgecolor='b', s=50,label='Dataset points')
    plt.plot([-3.0, 3.0], [-3.0*u1[1]/u1[0], 3.0*u1[1]/u1[0]], "k-", linewidth=1, label="PC1 Axis")
    plt.plot([-3.0, 3.0], [-3.0*u2[1]/u2[0], 3.0*u2[1]/u2[0]], "k--", linewidth=1,label="PC2 Axis")
    plt.text(u1[0] + 0.1, u1[1] - 0.05, r"$\mathbf{pc_1}$", fontsize=22)
    plt.text(u2[0] + 0.1, u2[1], r"$\mathbf{pc_2}$", fontsize=22)
    for axis, color in zip(U, ["red","green"]):
        start, end = X_mu, X_mu + 1 * axis
        ax.annotate('', xy=end, xytext=start,
                    arrowprops=dict(facecolor=color, width=2.0))
    ax.set_aspect('equal')
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    plt.axis([-3.0, 3.0, -3.0, 3.0])
    plt.legend()
    plt.grid(True)

def restore_data(Z, W):
    return Z.dot(W.T)

def plot_projected_data(X):
    plot = plt.scatter(X[:,0], X[:,1], s=50, facecolors='none', 
                       edgecolors='r',label='PCA Reduced Data Points')
    plt.grid(False)
    plt.tight_layout()
    plt.xlabel("$x_1$", fontsize=14)
    plt.ylabel("$x_2$", fontsize=14, rotation=0)
    plt.legend()

def visualize_projections(X_norm, X_rec, U):
    u1 = U[:,0].reshape(-1,1)
    u2 = U[:,1].reshape(-1,1)
    plot = plt.scatter(X_norm[:, 0], X_norm[:, 1], c='white', marker='o', 
                edgecolor='b', s=50,label='Original data points')

    plot = plt.scatter(X_rec[:,0], X_rec[:,1], c='white', marker='o', 
                       edgecolors='r', s=50, label='PCA Approximated 2D Data Points')
    
    plt.plot([-3.0, 3.0], [-3.0*u1[1]/u1[0], 3.0*u1[1]/u1[0]], "k--", linewidth=1, label="PC1 Axis")
    plt.xlabel("$x_1$", fontsize=14)
    plt.ylabel("$x_2$", fontsize=14, rotation=0)
    plt.axis([-3.0, 3.0, -3.0, 3.0])
    
    for x in range(X_norm.shape[0]):
        plt.plot([X_norm[x,0],X_rec[x,0]],[X_norm[x,1],X_rec[x,1]],'k--')
    plt.legend()
    plt.grid(True)


def main():
    print("\nProgramming Exercise 7: Principal Component Analysis\n")

    # Load dataset for PCA computation
    data    = sio.loadmat('data/ex7data1.mat')
    X = data['X']

    # Plot dataset
    input("\nPress <ENTER> key to plot dataset points ...")
    plt.figure(1)
    plot_data_points(X)
    plt.title("Dataset points")
    plt.show(block=False)
    #save_fig("PLOT_DATA")


    # PCA with SVD
    input("\nPress <ENTER> key to run PCA with SVD ...")
    X_norm,X_mu, X_std= feature_normalize(X)
    Sigma = compute_cov_matrix(X_norm)
    U,S,V = decompose_cov_matrix(Sigma)
    W,Z   = project_data(X_norm,U,k=1)
                           
    X_proj1 = X_norm.dot(U[:,0].reshape(-1,1)) 
    X_proj2 = X_norm.dot(U[:,1].reshape(-1,1)) 
    print('\nProjection of the first example is %0.3f.\n'%float(Z[0])) #Expected value of about 1.481
    print('Computed eigenvectors are:\n\n',U)
    print('\nTop principal component is',U[:,0])

    # Plot principal components
    plt.figure(2)
    input("\nPress <ENTER> key to plot principal components found ...")
    plot_eigen_vectors(X_norm,U)
    plt.title('Principal components axes')
    plt.show(block=False)
    #save_fig("PC_AXES")


    # Part 2: Reconstructing an approximation of the data
    input("\nPress <ENTER> to restore approximation of data ...")
    Xres = restore_data(Z, W)
    print('\nRecovered approximation of the first example is ',Xres[0])

    # Visualize projection of data
    input("\nPress <ENTER> key to visualize projections ...")
    plt.figure(3)
    visualize_projections(X_norm, Xres,U)
    plt.title('The normalized and projected data after PCA.')
    plt.show(block=False)
    #save_fig("PROJECTIONS")

    # Terminate program
    input("\nPress <ENTER> key to terminate program ...")


if __name__ == "__main__":
    main()
