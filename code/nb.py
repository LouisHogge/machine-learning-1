"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classification algorithms
"""
#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Documentation : https://scikit-learn.org/dev/developers/develop.html
# https://scikit-learn.org/stable/datasets/sample_generators.html
# https://scikit-learn.org/stable/modules/naive_bayes.html

import numpy as np
from matplotlib import pyplot as plt

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin

from sklearn.naive_bayes import GaussianNB # Already working Naives Bayesiens classifier -> Used to test
# https://github.com/scikit-learn/scikit-learn/blob/d99b728b3/sklearn/naive_bayes.py#L238
from sklearn.utils.multiclass import unique_labels

from data import make_dataset
from plot import plot_boundary

# (Question 3): Naive Bayes Classifier

class NaiveBayesClassifier(BaseEstimator, ClassifierMixin):
    def fit(self, X, y):
        """Fit a naive bayes classifier model using the training set
        (X, y).

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples.

        y : array-like, shape = [n_samples]
            The target values.

        Returns
        -------
        self : object
            Returns self.
        """
        # Input validation
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2 dimensional")

        # Requirement of the fit() method -> see documentation
        y = np.asarray(y)
        if y.shape[0] != X.shape[0]:
            raise ValueError("The number of samples differs between X and y")

        # ====================
        # TODO your code here.
        # ====================
        
        # Estimation of some parameters in the model is done on the training set and in the fit method.
        # The parameters are stored as attributes of the classifier object:
        
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        self.n_classes_ = len(self.classes_)
        self.n_features_ = X.shape[1]
        
        # Prior P(y) can be estimated from class frequencies in the learning set only.
        self.class_counts_ = np.zeros(self.n_classes_)
        self.class_prior_ = np.zeros(self.n_classes_)
        for i, c in enumerate(self.classes_):
            self.class_counts_[i] = np.sum(y == c)
            self.class_prior_[i] = self.class_counts_[i] / len(y)
        
        # Calculate mean and variance for each feature per class
        self.mean_ = np.zeros((self.n_classes_, self.n_features_))
        self.var_ = np.zeros((self.n_classes_, self.n_features_))
        for i, c in enumerate(self.classes_):
            X_c = X[y == c]
            self.mean_[i, :] = np.mean(X_c, axis=0)
            self.var_[i, :] = np.var(X_c, axis=0)
        
        # Returns the classifier
        return self
    
    def gaussian_density(self, x, mean, var):
        return (1 / np.sqrt(2 * np.pi * var)) * np.exp(-((x - mean) ** 2) / (2 * var))

    def predict(self, X):
        """Predict class for X.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        y : array of shape = [n_samples]
            The predicted classes, or the predict values.
        """

        # ====================
        # TODO your code here.
        # ====================

        prob = self.predict_proba(X)
        y = self.classes_[np.argmax(prob, axis=1)]
        return y

    def predict_proba(self, X):
        """Return probability estimates for the test data X.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        p : array of shape = [n_samples, n_classes]
            The class probabilities of the input samples. Classes are ordered
            by lexicographic order.
        """

        # ====================
        # TODO your code here.
        # ====================

        # Using the logarithmic maths then reverting back to normal scale with the exponent using the trick when returning.
        p = np.zeros((X.shape[0], self.n_classes_))
        for i in range(self.n_classes_):
            prior = np.log(self.class_prior_[i])
            log_p = np.sum(np.log(self.gaussian_density(X, self.mean_[i, :], self.var_[i, :])), axis=1)
            p[:, i] = prior + log_p
            
        # Normalize probabilities so they sum to 1 across classes
        p= np.exp(p)
        p /= p.sum(axis=1)[:, np.newaxis]
        return p # Here is a trick to get back the value using the logarithmic maths at first then converting to normal scale with the exponent.
    

    def predict_log_proba(self, X):
        """
        Predict logarithm of probability estimates.

        The returned estimates for all classes are ordered by the
        label of classes.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Vector to be scored, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        Returns
        -------
        T : array-like of shape (n_samples, n_classes)
            Returns the log-probability of the sample for each class in the
            model, where classes are ordered as they are in ``self.classes_``.
        """

        # ====================
        # TODO your code here.
        # ====================

        prob = self.predict_proba(X)
        T = np.log(prob)
        return T

if __name__ == "__main__":
    
    # Generating the Dataset
    n_points = 1500
    X, y = make_dataset(n_points,random_state=0)
    
    # Splitting the Dataset into 1200 as the training set and 300 as the test set
    X_train = X[:1200]
    y_train = y[:1200]
    X_test = X[1200:]
    y_test = y[1200:]
    
    """
    print("Length Xtrain " + str(len(X_train)))
    print("Length ytrain " + str(len(y_train)))
    print("Length Xtest " + str(len(X_test)))
    print("Length ytest " + str(len(y_test)))
    """
    
    # Training the classifier using the Gaussian Naive Bayes Classifier
    classifierGNB = GaussianNB()
    classifierGNB.fit(X_train, y_train)
    test_res = classifierGNB.predict(X_test)
    correct = y_test == test_res
    success_rate = np.count_nonzero(correct) / len(correct)
    # print("Success Rate: " + str(success_rate * 100) + "%")
    # Plotting the decision boundary
    # plot_boundary("naive_bayes_classifierGNB.pdf", classifierGNB, X, y, title="Naive Bayes")
    # Ensure that the test set is independent from the training set
    # plot_boundary("naive_bayes_test_classifierGNB.pdf", classifierGNB, X_test, y_test, title="Naive Bayes Test")
    
    # Our own Naive Bayes Classifier is expected to have the same results as the Gaussian Naive Bayes Classifier.
    Nbg = NaiveBayesClassifier()
    Nbg.fit(X_train, y_train)
    test_res2 = Nbg.predict(X_test)
    correct2 = y_test == test_res2
    success_rate2 = np.count_nonzero(correct2) / len(correct2)
    # print("Success Rate: " + str(success_rate2 * 100) + "%")
    # Plotting the decision boundary
    plot_boundary("naive_bayes_nbg.pdf", Nbg, X, y, title="Naive Bayes")
    # Ensure that the test set is independent from the training set
    plot_boundary("naive_bayes_test_nbg.pdf", Nbg, X_test, y_test, title="Naive Bayes Test")
    
    # Number of generations
    num_generations = 5
    accuracies = []

    for _ in range(num_generations):
        # Generating the Dataset
        n_points = 1500
        seed = _
        X, y = make_dataset(n_points, random_state=seed)
        
        # Splitting the Dataset into 1200 as the training set and 300 as the test set
        X_train = X[:1200]
        y_train = y[:1200]
        X_test = X[1200:]
        y_test = y[1200:]
        
        # Training the classifier using your Naive Bayes Classifier
        Nbg = NaiveBayesClassifier()
        Nbg.fit(X_train, y_train)
        test_res = Nbg.predict(X_test)
        
        # Calculate accuracy
        correct = y_test == test_res
        success_rate = np.count_nonzero(correct) / len(correct)
        accuracies.append(success_rate)

    # Compute the average and standard deviation
    average_accuracy = np.mean(accuracies)
    std_deviation = np.std(accuracies)

    print("Average Accuracy: {:.2f}%".format(average_accuracy * 100))
    print("Standard Deviation: {:.2f}%".format(std_deviation * 100))