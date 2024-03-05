"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classification algorithms
"""
#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from data import make_dataset
from plot import plot_boundary

def build_fit_predict_knn(n_neighbors, x_learning, y_learning, x_test, y_test):
    """
    Build, fit, and evaluate a k-Nearest Neighbors classifier.

    Parameters:
    - n_neighbors (int): number of neighbors to use.
    - x_learning (numpy.ndarray): features of the learning set. Shape (n_samples, n_features).
    - y_learning (numpy.ndarray): outputs of the learning set. Shape (n_samples,).
    - x_test (numpy.ndarray): features of the test set. Shape (n_samples, n_features).
    - y_test (numpy.ndarray): outputs of the test set. Shape (n_samples,).

    Returns:
    - knn_model (KNeighborsClassifier): trained knn model.
    - accuracy (float): accuracy of the model on the test set.
    """
    
    # build knn model with given n_neighbors
    knn_model = KNeighborsClassifier(n_neighbors=n_neighbors)
    
    # fit knn model on learning set
    knn_model.fit(x_learning, y_learning)
    
    # prediction using knn model on test set
    y_predict = knn_model.predict(x_test)
    
    # accuracy of knn model
    accuracy = accuracy_score(y_test, y_predict)
    
    return knn_model, accuracy

if __name__ == "__main__":
    # learning set size: 1200, test set size: 300
    nb_samples = 1500
    learning_set_size = 1200
    
    n_neighbors_values = [1, 5, 25, 125, 625, 1200]
    
    accuracies = {n_neighbors: [] for n_neighbors in n_neighbors_values}
    
    # 5 generations of dataset
    for i in range(5):

        # dataset
        X, y = make_dataset(nb_samples, random_state=0+i)

        # learning set
        x_learning, y_learning = X[:learning_set_size], y[:learning_set_size]

        # test set
        x_test, y_test = X[learning_set_size:], y[learning_set_size:]
        
        for n_neighbors in n_neighbors_values:
            
            knn_model, accuracy = build_fit_predict_knn(n_neighbors, x_learning, y_learning, x_test, y_test)

            accuracies[n_neighbors].append(accuracy)
            
            # plot classification boundary
            plot_boundary(f'knn_k_{n_neighbors}_gen_{i}', knn_model, x_test, y_test, title=f"knn (k={n_neighbors}, generation={i})")
    
    # print average accuracies and standard deviations
    for n_neighbors in n_neighbors_values:
        avg_accuracy = np.mean(accuracies[n_neighbors])

        std_accuracy = np.std(accuracies[n_neighbors])

        print(f"Number of Neighbors: {n_neighbors}, Average Accuracy: {avg_accuracy:.4f}, Standard Deviation: {std_accuracy:.4f}")
