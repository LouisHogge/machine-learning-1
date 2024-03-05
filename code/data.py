"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classification algorithms
"""
import numpy as np
from sklearn.utils import check_random_state
from sklearn.datasets import make_moons


def make_dataset(n_points, random_state=None):
    """Generate a two-moon 2D dataset

    Parameters
    -----------
    n_points : int >0
        The number of points to generate
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Return
    ------
    X : array of shape [n_points, 2]
        The feature matrix of the dataset
    y : array of shape [n_points]
        The labels of the dataset
    """
    random_state = check_random_state(random_state)
    
    X, y = make_moons(n_samples=n_points, noise=0.2, random_state=random_state)

    X = 2 * X

    theta = np.pi/6
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], \
                                [np.sin(theta), np.cos(theta)]])
    X = np.dot(X, rotation_matrix)

    permutation = np.arange(n_points)
    random_state.shuffle(permutation)
    return X[permutation], y[permutation]


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    n_points = 1500
    plt.figure()
    X, y = make_dataset(n_points)
    X_y0 = X[y==0]
    X_y1 = X[y==1]
    plt.scatter(X_y0[:,0], X_y0[:,1], color="DodgerBlue", alpha=.5)
    plt.scatter(X_y1[:,0], X_y1[:,1], color="orange", alpha=.5)
    plt.grid(True)
    plt.xlabel("X_0")
    plt.ylabel("X_1")
    plt.xlim([-6, 6])
    plt.ylim([-6, 6])
    plt.title("Two-moon dataset")
    plt.savefig("two_moon.pdf")
