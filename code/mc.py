"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classification algorithms
"""
#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt

from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from data import make_dataset
from plot import plot_boundary

# (Question 4): Methods Comparison

def dt_tuning(x_learning, y_learning, x_test, y_test):
    """
    Tune the hyperparameters of a Decision Tree classifier using GridSearchCV and evaluate its performance on the test set.

    Parameters:
    - x_learning (numpy.ndarray): features of the learning set. Shape (n_samples, n_features).
    - y_learning (numpy.ndarray): outputs of the learning set. Shape (n_samples,).
    - x_test (numpy.ndarray): features of the test set. Shape (n_samples, n_features).
    - y_test (numpy.ndarray): outputs of the test set. Shape (n_samples,).

    Returns:
    - best_max_depth (int or None): the optimal maximum depth of the tree found by GridSearchCV.
    - accuracy (float): accuracy of the Decision Tree model on the test set.
    """

    # tune hyperparameters for decision tree
    dt_model = GridSearchCV(estimator=DecisionTreeClassifier(random_state=0), param_grid={'max_depth': [1, 2, 4, 8, None]}, cv=5, scoring='accuracy')
    dt_model.fit(x_learning, y_learning)
    best_max_depth = dt_model.best_params_['max_depth']
    
    # prediction using decision tree model on test set
    y_predict = dt_model.predict(x_test)
    
    # accuracy of decision tree model
    accuracy = accuracy_score(y_test, y_predict)

    return best_max_depth, accuracy

def knn_tuning(x_learning, y_learning, x_test, y_test):
    """
    Tune the hyperparameters of a k-Nearest Neighbors classifier using GridSearchCV and evaluate its performance on the test set.

    Parameters:
    - x_learning (numpy.ndarray): features of the learning set. Shape (n_samples, n_features).
    - y_learning (numpy.ndarray): outputs of the learning set. Shape (n_samples,).
    - x_test (numpy.ndarray): features of the test set. Shape (n_samples, n_features).
    - y_test (numpy.ndarray): outputs of the test set. Shape (n_samples,).

    Returns:
    - best_n_neighbors (int): the optimal number of neighbors found by GridSearchCV.
    - accuracy (float): accuracy of the k-NN model on the test set.
    """

    # tune hyperparameters for knn
    knn_model = GridSearchCV(estimator=KNeighborsClassifier(), param_grid={'n_neighbors': [1, 5, 25, 125, 625]}, cv=5, scoring='accuracy')
    knn_model.fit(x_learning, y_learning)
    best_n_neighbors = knn_model.best_params_['n_neighbors']
    
    # prediction using knn model on test set
    y_predict = knn_model.predict(x_test)
    
    # accuracy of knn model
    accuracy = accuracy_score(y_test, y_predict)

    return best_n_neighbors, accuracy

if __name__ == "__main__":
    dt_best_max_depths = []
    knn_best_n_neighbors = []


    # learning set size: 1200, test set size: 300
    nb_samples = 1500
    learning_set_size = 1200

    dt_accuracies = []
    knn_accuracies = []
    
    # 5 generations of dataset
    for i in range(5):

        # dataset
        X, y = make_dataset(nb_samples, random_state=0+i)

        # learning set
        x_learning, y_learning = X[:learning_set_size], y[:learning_set_size]

        # test set
        x_test, y_test = X[learning_set_size:], y[learning_set_size:]

        # run decision tree experiment
        best_max_depth, dt_accuracy = dt_tuning(x_learning, y_learning, x_test, y_test)
        dt_accuracies.append(dt_accuracy)
        dt_best_max_depths.append(best_max_depth)

        # run k-nearest neighbors experiment
        best_n_neighbors, knn_accuracy = knn_tuning(x_learning, y_learning, x_test, y_test)
        knn_accuracies.append(knn_accuracy)
        knn_best_n_neighbors.append(best_n_neighbors)

    # calculate average accuracies and standard deviations
    dt_avg_accuracy = np.mean(dt_accuracies)
    dt_std_accuracy = np.std(dt_accuracies)
    knn_avg_accuracy = np.mean(knn_accuracies)
    knn_std_accuracy = np.std(knn_accuracies)
    
    # print results
    print(f"Decision Tree - Average Test Accuracy: {dt_avg_accuracy:.4f}, Standard Deviation: {dt_std_accuracy:.4f}")
    print(f"k-NN - Average Test Accuracy: {knn_avg_accuracy:.4f}, Standard Deviation: {knn_std_accuracy:.4f}")

    print(f"Decision Tree - Best Max Depth for each generation: {dt_best_max_depths}")
    print(f"k-NN - Best N Neighbors for each generation: {knn_best_n_neighbors}")
