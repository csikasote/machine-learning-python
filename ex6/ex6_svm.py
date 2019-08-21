# Python ≥3.5 is required
import sys
assert sys.version_info >= (3, 5)

# Common imports
import scipy.io as sio
import numpy as np
import os

# sklearn imports for SVM
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC

# to make this notebook's output stable across runs
np.random.seed(42)

# To plot pretty figures
import matplotlib
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


def plot_dataset(X, y):
    #Find Indices of Positive and Negative Examples
    pos = (y == 1).nonzero(); 
    neg = (y == 0).nonzero();
    #plot the data
    plt.plot(X[pos, 0], X[pos, 1],'bs',linewidth=4, markersize=5, label='Positive');
    plt.plot(X[neg, 0], X[neg, 1],'g^',linewidth=4, markersize=5, label='Negative');
    plt.xlabel(r"$x_1$", fontsize=20)
    plt.ylabel(r"$x_2$", fontsize=20, rotation=0)
    plt.grid(True)

def svm_linear_boundary(model,X,y, ax= None):
    w = model.coef_[0];
    b = model.intercept_[0];
    
    # At the decision boundary, w0*x0 + w1*x1 + b = 0
    # => x1 = -w0/w1 * x0 - b/w1
    xmin = X.min(); xmax = X.max();
    xp = np.linspace(xmin, xmax, 200)
    yp = -w[0]/w[1] * xp - b/w[1]
    
    margin = 1/w[1]
    gutter_up = yp + margin
    gutter_down = yp - margin
    
    svs = model.support_vectors_
    plt.scatter(svs[:, 0], svs[:, 1], s=180, linewidth=8, facecolors='#FFAAAA')
    plt.plot(xp, yp, "k-", linewidth=2)
    plt.plot(xp, gutter_up, "k--", linewidth=2)
    plt.plot(xp, gutter_down, "k--", linewidth=2)
    plt.xlabel(r"$x_1$", fontsize=20)
    plt.ylabel(r"$x_2$", fontsize=20, rotation=0)


def plot_linear_svm(model,X,y):
    #g = plt.figure(2)
    plot_dataset(X, y)
    svm_linear_boundary(model,X,y)
    plt.annotate("Outlier",
                 xy=(0.1, 4.1),
                 xytext=(0.4, 3.0),
                 ha="center",
                 arrowprops=dict(facecolor='black', width=2, shrink=0.1),
                 fontsize=14,
                )
    plt.title('SVC with Linear Kernel (C = %d)'%(1))
    plt.axis([0,5,1.5,5])

def plot_svm_dataset2(clf,X,y):
    plot_dataset(X, y)
    x0s = np.linspace(X.min()-0.05, X.max()+0.05, 100)
    x1s = np.linspace(y.min()+0.38, y.max()+0.01, 100)

    x0, x1 = np.meshgrid(x0s, x1s)
    X = np.c_[x0.ravel(), x1.ravel()]   
    y_pred = clf.predict(X).reshape(x0.shape)
    y_decision = clf.decision_function(X).reshape(x0.shape)
    
    plt.title("SVC with Gaussian Kernel", fontsize=18)
    plt.contourf(x0, x1, y_pred, cmap=plt.cm.brg, alpha=0.2)
    plt.contourf(x0, x1, y_decision, cmap=plt.cm.brg, alpha=0.1)

def plot_svm_dataset3(clf,X,y):
    plot_dataset(X, y)
    x0s = np.linspace(X.min(), X.max()-0.25, 100)
    x1s = np.linspace(y.min()-0.7, y.max()-0.4, 100)
    x0, x1 = np.meshgrid(x0s, x1s)
    X = np.c_[x0.ravel(), x1.ravel()]
    
    y_pred = clf.predict(X).reshape(x0.shape)
    y_decision = clf.decision_function(X).reshape(x0.shape)

    plt.contourf(x0, x1, y_pred, cmap=plt.cm.brg, alpha=0.2)
    plt.contourf(x0, x1, y_decision, cmap=plt.cm.brg, alpha=0.1)


def cross_validation(X3,y3,Xval,yval,val_range):
    pair  = (0, 0) 
    score = 0
    for c in val_range:
        for sigma in val_range:
            gamma = 0.5 * sigma**(-2);
            rbf_kernel_svm_clf = Pipeline([
                    ("scaler", StandardScaler()),
                    ("svm_clf", SVC(kernel="rbf", gamma=gamma, C=c))
                ])
            rbf_model = rbf_kernel_svm_clf.fit(X3, y3)
            cur_score = rbf_model.score(Xval,yval)
            if cur_score > score:
                score = cur_score
                pair = (c, sigma)
    return pair[0],pair[1],score
    

def main():
    # Load dataset for part 1
    dataset1    = sio.loadmat('data/ex6data1.mat')
    X1       = dataset1["X"];
    y1       = dataset1["y"].ravel();

    # Plot training dataset 1
    input("\nPress <ENTER> to plot dataset 1 ...")
    plt.figure(1)
    plot_dataset(X1, y1)
    plt.title("Training dataset 1", fontsize=18)
    plt.axis([0,5,1.5,5])
    plt.show(block=False)
    save_fig("TD1")

    # Setting SVM for classification
    C = 1.0  # SVM regularization parameter
    svm_linear_kernel_model = SVC(kernel="linear", C=C)
    svm_linear_kernel_model = svm_linear_kernel_model.fit(X1, y1)

    # Plot SVM classification
    input("\nPress <ENTER> to plot SVM classification for dataset 1 ...")
    plt.figure(2)
    plot_linear_svm(svm_linear_kernel_model,X1,y1)
    plt.show(block=False)
    save_fig("SVM_RESULT_TD1")

    # Part 2: SVM for Nonlinear Classification
    dataset2 = sio.loadmat('data/ex6data2.mat')
    X2       = dataset2["X"];
    y2       = dataset2["y"].ravel()

    # Plot training dataset 2
    input("\nPress <ENTER> to plot dataset 2 ...")
    plt.figure(3)
    plot_dataset(X2, y2)
    plt.ylim([0.3,1])
    plt.title("Example Dataset 2", fontsize=18)
    plt.show(block=False)
    save_fig("TD2")

    # SVM for Non-Linear Classification
    sigma = 0.1           # sigma component of the gaussian kernel formula
    gamma = 0.5 * sigma**-2 # Computing gamma for gaussian kernal
    rbf_kernel_svm_clf = Pipeline([
            ("scaler", StandardScaler()),
            ("svm_clf", SVC(kernel="rbf", gamma=gamma, C=1.0))
        ])
    rbf_clf = rbf_kernel_svm_clf.fit(X2, y2)  

    # Plot SVM classification for Nonlinear
    input("\nPress <ENTER> to plot SVM classification of dataset 2 ...")
    plt.figure(4)
    plot_svm_dataset2(rbf_clf,X2,y2)
    plt.show(block=False)
    save_fig("SVM_RESULT_TD2")


    # Training dataset 3
    dataset3 = sio.loadmat('data/ex6data3.mat' )
    X3       = dataset3["X"];
    y3       = dataset3["y"].ravel();
    Xval     = dataset3["Xval"];
    yval     = dataset3["yval"].ravel();


    # Plot dataset 3
    input("\nPress <ENTER> to plot dataset 3 ...")
    plt.figure(5)
    plt.title("Training dataset 3")
    plot_dataset(X3, y3)
    plt.show(block=False)
    save_fig("TD2")


    # Cross validation on dataset3
    input("\nPress <ENTER> key to run cross validation on dataset3 ...")
    val_range = [0.01, 0.03, 0.1, 0.3, 1., 3., 10., 30.]
    C, sigma,score = cross_validation(X3,y3,Xval,yval,val_range)
    print('\nCross-validation completed ....')
    print("\nBest (C, sigma) pair found by cross-validation is (C = %.2f, sigma = %.2f) with a score of %.4f.\n"%\
      (C,sigma,score))

    # Run SVM with results from CV
    gamma = 0.5 * sigma**-2;
    rbf_kernel_svm_clf = Pipeline([
            ("scaler", StandardScaler()),
            ("svm_clf", SVC(kernel="rbf", gamma=gamma, C=C))
        ])
    rbf_model = rbf_kernel_svm_clf.fit(X3, y3)

    # Plot SVM classification boundary
    input("\nPress <ENTER> to visualize SVM decision boundary ...")
    plt.figure(6)
    plt.title("SVC with Gaussian Kernel", fontsize=18)
    plot_svm_dataset3(rbf_model, X3,y3)
    plt.show(block=False)
    save_fig("SVM_RESULT_TD3")

    # Terminate program
    input("\nPress <ENTER> key to terminate process ...")


if __name__ == "__main__":
    print("\nProgramming Exercise 6: Support Vector Machines\n")
    main()
