"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classification algorithms
"""
#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from pylab import *

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

from data import make_data1, make_data2
from plot import plot_boundary


# (Question 2)
def KNN(X, y, n_neighbors, plot_name, plot_title):
    # K-nearest neighbors Classifier
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    # Fitting
    knn.fit(X,y)
    # Plot
    p = plot_boundary(plot_name, knn, X[0:150], y[0:150], title=plot_title)
    #Return fitted estimator (decision tree)
    return knn

def fold_cross(n_neighbors, cv):
    # Create dataset 2 with seed 0
    X2,y2 = make_data2(2000, random_state=0)
    # K-nearest neighbors Classifier
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    # 10-Fold cross validation strategy
    scores = cross_val_score(knn, X2, y2, cv=cv)
    return scores.mean(), scores.std()

if __name__ == "__main__":
    # Create dataset 1 with seed 0
    X1,y1 = make_data1(2000, random_state=0)
    # K-nearest neighbors Classifier
    knn1_1 = KNN(X1[0:150], y1[0:150], 1, "KNN1_1", "")
    knn1_5 = KNN(X1[0:150], y1[0:150], 5, "KNN1_5", "")
    knn1_10 = KNN(X1[0:150], y1[0:150], 10, "KNN1_10", "")
    knn1_75 = KNN(X1[0:150], y1[0:150], 75, "KNN1_75", "")
    knn1_100 = KNN(X1[0:150], y1[0:150], 100, "KNN1_100", "")
    knn1_150 = KNN(X1[0:150], y1[0:150], 150, "KNN1_150", "")

    # Create dataset 2 with seed 0
    X2,y2 = make_data2(2000, random_state=0)
    # K-nearest neighbors Classifier
    knn2_1 = KNN(X2[0:150], y2[0:150], 1, "KNN2_1", "")
    knn2_5 = KNN(X2[0:150], y2[0:150], 5, "KNN2_5", "")
    knn2_10 = KNN(X2[0:150], y2[0:150], 10, "KNN2_10", "")
    knn2_75 = KNN(X2[0:150], y2[0:150], 75, "KNN2_75", "")
    knn2_100 = KNN(X2[0:150], y2[0:150], 100, "KNN2_100", "")
    knn2_150 = KNN(X2[0:150], y2[0:150], 150, "KNN2_150", "")

    # 10-Fold cross validation strategy
    acc = [0] * 149
    std = [0] * 149
    # Compute final accuracy using 10-fold cross validation
    # for each number of neighbors
    for k in range(1, 150):
        acc[k-1], std[k-1] = fold_cross(k,10)

    # Transforma acc to np array
    acc_plot = np.array(acc)

    # Find min and max accuracy
    min = np.amin(acc_plot)
    max = np.amax(acc_plot)
    max_index = np.where(acc_plot == max)
    print('Max = ' + str(max))
    print('Max index = ' + str(max_index[0]+1))

    # Bar plot of accuracies
    plt.bar(range(1,150), acc_plot - min, width=1, bottom = min, align='edge', linewidth = 0)

    # Highlight max
    plt.bar(max_index[0]+1, max - min, width=1, bottom = min, align='edge', linewidth = 0, color ='red')
    plt.xlabel('Number of neighbors')
    plt.ylabel('Final accuracy of ten-fold cross validation')
    plt.show()
