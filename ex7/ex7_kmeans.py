# Python ≥3.5 is required
import sys
assert sys.version_info >= (3, 5)

# Scikit-Learn ≥0.20 is required
import sklearn
assert sklearn.__version__ >= "0.20"

# K-measn algorithm
from sklearn.cluster import KMeans

# Common imports
import scipy.io as sio
import numpy as np
import os

# To plot pretty figures
import matplotlib as mpl
import matplotlib.pyplot as plt

# To read images 
from matplotlib.image import imread

# Function to plot datapoints 
def plot_data_points(X, y=None):
    plt.scatter(X[:, 0], X[:, 1], 
                c='white', marker='o', 
                edgecolor='black', s=50,label='Training data points')
    plt.grid()
    plt.tight_layout()
    plt.xlabel("$x_1$", fontsize=14)
    plt.ylabel("$x_2$", fontsize=14, rotation=0)
    plt.axis([-2, 10,0,6])
    plt.legend()

# Function to plot the centroids
def plot_centroids(centroids, weights=None, circle_color='w', cross_color='k'):
    plt.scatter(centroids[:, 0],
                centroids[:, 1], 
                s=250, marker='*',
                c='red', 
                edgecolor='black',
                label='centroids')
    plt.legend()

def plot_clusters(X,y):
    plt.scatter(X[y == 0, 0],X[y == 0, 1],
                s=50, c='lightgreen',
                marker='s', edgecolor='black',label='cluster 1')
    plt.scatter(X[y == 1, 0],X[y == 1, 1],
                s=50, c='orange',
                marker='o', edgecolor='black',
                label='cluster 2')
    plt.scatter(X[y == 2, 0],
                X[y == 2, 1],
                s=50, c='lightblue',
                marker='v', edgecolor='black',
                label='cluster 3')
    plt.legend(scatterpoints=1)
    plt.tight_layout()
    plt.xlabel("$x_1$", fontsize=14)
    plt.ylabel("$x_2$", fontsize=14, rotation=0)
    plt.axis([-2, 10,0,6])

# Plot the decision boundary
def plot_decision_boundaries(cluster_model, X, resolution=1000, show_centroids=True,
                             show_xlabels=True, show_ylabels=True):
    mins = X.min(axis=0) - 0.1
    maxs = X.max(axis=0) + 0.1
    xx, yy = np.meshgrid(np.linspace(mins[0], maxs[0], resolution),
                         np.linspace(mins[1], maxs[1], resolution))
    Z = cluster_model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]),
                cmap="Pastel2")
    plt.contour(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]),
                linewidths=1, colors='k')
    plt.scatter(X[:, 0], X[:, 1], 
                c='white', marker='o', 
                edgecolor='black', s=50)
    if show_centroids:
        plot_centroids(cluster_model.cluster_centers_)

    if show_xlabels:
        plt.xlabel("$x_1$", fontsize=14)
    else:
        plt.tick_params(labelbottom=False)
    if show_ylabels:
        plt.ylabel("$x_2$", fontsize=14, rotation=0)
    else:
        plt.tick_params(labelleft=False)

def km_elbow_method(X):
    distortions = []
    for k in range(1, 13):
        kmeans = KMeans(n_clusters=k, 
                    init='k-means++', 
                    n_init=10, 
                    max_iter=300, 
                    random_state=42)
        kmeans.fit(X)
        distortions.append(kmeans.inertia_)
    return distortions

def plot_elbow(distortions): 
    plt.plot(range(1, 13), distortions, marker='o', color='g', markersize=10)
    plt.grid()
    plt.xlabel('# of clusters')
    plt.ylabel('Distortion')
    plt.tight_layout
    plt.axis([0,12, 0,2000])


def plot_annotated_elbow(distortions): 
    plt.plot(range(1, 13), distortions, marker='o', color='g',markersize=10)
    plt.grid()
    plt.xlabel('# of clusters')
    plt.ylabel('Distortion')
    plt.annotate('Elbow (k = 3)',
                 xy=(3, distortions[2]),
                 xytext=(0.75, 0.45),
                 textcoords='figure fraction',
                 fontsize=16,
                 arrowprops=dict(facecolor='blue', shrink=0.1)
                )
    plt.tight_layout()
    plt.axis([0,12, 0,2000])


# PART 2
def visualize_image(img_path):
    original_img = imread(img_path)
    plt.imshow(original_img)
    plt.title('Original image')
    plt.axis('off')

def preprocess_image(img_path):
    original_img = imread(img_path) # Load the image for procession

    # Load Image and transform to a 2D numpy array.
    w, h, d = original_shape = tuple(original_img.shape)
    assert d == 3
    
    # We need an (n_sample, n_feature) array
    processed_img = np.reshape(original_img, (w * h, d)) 
    
    return original_img, processed_img

def train_kmeans(n_colors, original_img, processed_img):
    
    k_means = KMeans(n_clusters=n_colors, random_state=42).fit(processed_img)
    
    # create an array from labels and values
    labels = k_means.labels_
    compressed_img = k_means.cluster_centers_[labels]
    compressed_img = compressed_img.reshape(original_img.shape)
    
    return compressed_img


def main():
    print("\nProgramming Exercise 7: KMeans Algorithm\n")
    # Load dataset for kmeans algorithm
    data    = sio.loadmat('data/ex7data2.mat')
    X = data['X']

    # Plot dataset points
    input("\nPress <ENTER> to plot dataset points for KMeans algorithm ...")
    plt.figure(1)
    plot_data_points(X, y=None)
    plt.title('Training data point')
    plt.show(block=False)

    # Initial set of centroids
    K = 3; # Three Centroids
    initial_centroids = np.array([[3,3],[6,2],[8,5]]);

    # Instantiating a KMeans object
    km_model = KMeans(n_clusters=K,           # n_clusters = 3
                  init=initial_centroids,
                  n_init=1,
                  random_state=42)

    # Fit the Kmeans algorithm
    y_pred         = km_model.fit_predict(X) 
    centroids = km_model.cluster_centers_    
    print('\nCentroids computed after initial finding of closest centroids: \n', centroids)

    # Plot the computed centroids
    plt.figure(2)
    plot_data_points(X)
    plot_centroids(centroids)
    plt.title('Dataset points & computed centroids')
    plt.show(block=False)

    #Plot the cluster plot decision boundary
    plt.figure(3)
    plot_decision_boundaries(km_model, X)
    plt.title('Computed K-Means decision boundary')
    plt.show(block=False)

    # Plot computed clusters
    plt.figure(4)
    plot_clusters(X,y_pred)
    plot_centroids(centroids)
    plt.title('Computed K-Means clusters')
    plt.grid(True)
    plt.show(block=False)

    # Evaluating kmeans
    print('Distortion: %.2f' % km_model.inertia_)

    X_dist = km_model.transform(X) # Measure the distance from each datapoint instance to every centroid
    distortion = np.sum(X_dist[np.arange(len(X_dist)), km_model.labels_]**2)
    print('Distortion: %.2f' % distortion)

    # Compute distortions 
    distortions = km_elbow_method(X)

    # Plot distortions vs number of clusters
    plt.figure(5)
    plot_elbow(distortions)
    plt.title('Distortions vs. number of clusters')
    plt.show(block=False)

    # Plot Annotation of the elbow on the graph
    plt.figure(6)
    plot_annotated_elbow(distortions)
    plt.title('Annotation of the elbow')
    plt.show(block=False)

    # PART 2: Image compression with K-Means Algorithm
    img_path = os.path.join("./images/","bird_small.png")

    plt.figure(7)
    visualize_image(img_path)
    plt.show(block=False)

    original_img, processed_img = preprocess_image(img_path)
    print('The shape of the original image is ' + str(original_img.shape)+'\n')
    print('The shape of the preprocessed image is ' + str(processed_img.shape))

    # Training the K means algorithm to compress a single image
    compressed_img = train_kmeans(16, original_img, processed_img)
    print('The shape of compressed image is ', compressed_img.shape)

    # Plot the pictures
    plt.figure(8)
    plt.imshow(compressed_img)
    plt.title('Compressed image')
    plt.axis('off')
    plt.show(block=False)

    # Terminate program
    input("Press <ENTER> key to terminate program ...")

if __name__ == "__main__":
    main()
