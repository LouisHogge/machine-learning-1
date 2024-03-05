"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classification algorithms
"""
#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

from data import make_dataset
from plot import plot_boundary, make_cmaps

def plot_dataset(fname, X, y, title=""):
    """
    Plot and save a scatter plot of a dataset with 2D features.

    Parameters:
    - fname (str): the filename to save the plot.
    - X (numpy.ndarray): features of the dataset. Shape (n_samples, n_features).
    - y (numpy.ndarray): outputs of the dataset. Shape (n_samples,).
    - title (str, optional): the title of the plot. Default is an empty string.

    Returns:
    - None
    """

    _, sc_map = make_cmaps()

    plt.figure()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title(title)
    plt.xlabel("X_0")
    plt.ylabel("X_1")

    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=sc_map, edgecolor='black', s=10)
    plt.xlim(X[:, 0].min() - .5, X[:, 0].max() + .5)
    plt.ylim(X[:, 1].min() - .5, X[:, 1].max() + .5)

    plt.savefig('{}.pdf'.format(fname), transparent=True)
    plt.close()

# (Question 1): Decision Trees

def build_fit_predict_dt(max_depth, x_learning, y_learning, x_test, y_test):
    """
    Build, fit, and evaluate a Decision Tree classifier.

    Parameters:
    - max_depth (int or None): maximum depth of the tree. If None, nodes are expanded until all leaves are pure.
    - x_learning (numpy.ndarray): features of the learning set. Shape (n_samples, n_features).
    - y_learning (numpy.ndarray): outputs of the learning set. Shape (n_samples,).
    - x_test (numpy.ndarray): features of the test set. Shape (n_samples, n_features).
    - y_test (numpy.ndarray): outputs of the test set. Shape (n_samples,).

    Returns:
    - dt_model (DecisionTreeClassifier): trained decision tree model.
    - accuracy (float): accuracy of the model on the test set.
    """

    # build decision tree model with given max_depth
    dt_model = DecisionTreeClassifier(max_depth=max_depth, random_state=0)
    
    # fit decision tree model on learning set
    dt_model.fit(x_learning, y_learning)
    
    # prediction using decision tree model on test set
    y_predict = dt_model.predict(x_test)
    
    # accuracy of decision tree model
    accuracy = accuracy_score(y_test, y_predict)
    
    return dt_model, accuracy

if __name__ == "__main__":
    # learning set size: 1200, test set size: 300
    nb_samples = 1500
    learning_set_size = 1200
    
    max_depth_values = [1, 2, 4, 8, None]
    
    accuracies = {depth: [] for depth in max_depth_values}
    
    # 5 generations of dataset
    for i in range(5):

        # dataset
        X, y = make_dataset(nb_samples, random_state=0+i)

        # learning set
        x_learning, y_learning = X[:learning_set_size], y[:learning_set_size]

        # test set
        x_test, y_test = X[learning_set_size:], y[learning_set_size:]

        # plot test set
        plot_dataset(f'test_generation_{i}', x_test, y_test, title=f'Test Generation {i}')
        
        for depth in max_depth_values:
            
            dt_model, accuracy = build_fit_predict_dt(depth, x_learning, y_learning, x_test, y_test)

            accuracies[depth].append(accuracy)
            
            # plot classification boundary
            plot_boundary(f'dt_depth_{depth}_gen_{i}', dt_model, x_test, y_test, title=f"Decision Tree (max_depth={depth}, generation={i})")
    
    # print average accuracies and standard deviations
    for depth in max_depth_values:
        avg_accuracy = np.mean(accuracies[depth])

        std_accuracy = np.std(accuracies[depth])

        print(f"Max Depth: {depth}, Average Accuracy: {avg_accuracy:.4f}, Standard Deviation: {std_accuracy:.4f}")
