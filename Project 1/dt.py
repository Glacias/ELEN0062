"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classification algorithms
"""
#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib
import matplotlib.pyplot as plt


from sklearn.tree import DecisionTreeClassifier

from data import make_data1, make_data2
from plot import plot_boundary


# (Question 1)
def DT(X, y, max_d, plot_name, plot_title):
    # Decision Tree Classifier
    dt = DecisionTreeClassifier(max_depth=max_d)
    # Fitting
    dt.fit(X,y)
    # Plot
    p = plot_boundary(plot_name, dt, X[0:150], y[0:150], title=plot_title)
    #Return fitted estimator (decision tree)
    return dt

def Avg_std_acc_d1(max_depth, nbr_gen):
    #Create a list of accuracy
    acc = []
    #Compute test set accuracy on max_depth
    for k in range(nbr_gen):
        # Create dataset 1 with seed k (0 to nbr_gen-1)
        X,y = make_data1(2000, random_state=k)
        # Decision Tree Classifier
        dt = DecisionTreeClassifier(max_depth=max_depth)
        # Fitting
        dt.fit(X[0:150], y[0:150])
        # Test accuracy
        acc.append(dt.score(X[150:2000], y[150:2000]))
    #Compute average and standard deviation
    return np.mean(acc), np.std(acc)

def Avg_std_acc_d2(max_depth, nbr_gen):
    #Create a list of accuracy
    acc = []
    #Compute test set accuracy on max_depth
    for k in range(nbr_gen):
        # Create dataset 2 with seed k (0 to nbr_gen-1)
        X,y = make_data2(2000, random_state=k)
        # Decision Tree Classifier
        dt = DecisionTreeClassifier(max_depth=max_depth)
        # Fitting
        dt.fit(X[0:150], y[0:150])
        # Test accuracy
        acc.append(dt.score(X[150:2000], y[150:2000]))
    #Compute average and standard deviation
    return np.mean(acc), np.std(acc)

if __name__ == "__main__":

    # Create dataset 1 with seed 0
    X,y = make_data1(2000, random_state=0)
    # Decision Tree Classifier
    dt1_1 = DT(X[0:150], y[0:150], 1, "DT1_1", "")
    dt1_2 = DT(X[0:150], y[0:150], 2, "DT1_2", "")
    dt1_4 = DT(X[0:150], y[0:150], 4, "DT1_4", "")
    dt1_8 = DT(X[0:150], y[0:150], 8, "DT1_8", "")
    dt1_unb = DT(X[0:150], y[0:150], None, "DT1_unb", "")

    # Create dataset 2 with seed 0
    X2,y2 = make_data2(2000, random_state=0)
    # Decision Tree Classifier
    dt2_1 = DT(X2[0:150], y2[0:150], 1, "DT2_1", "")
    dt2_2 = DT(X2[0:150], y2[0:150], 2, "DT2_2", "")
    dt2_4 = DT(X2[0:150], y2[0:150], 4, "DT2_4", "")
    dt2_8 = DT(X2[0:150], y2[0:150], 8, "DT2_8", "")
    dt2_unb = DT(X2[0:150], y2[0:150], None, "DT2_unb", "")

    #### Plot ####

    labels = ['Depth 1', 'Depth 2', 'Depth 4', 'Depth 8', 'Unbounded']

    avg1 = [0] * 5
    std1 = [0] * 5
    avg2 = [0] * 5
    std2 = [0] * 5

    avg1[0], std1[0] = Avg_std_acc_d1(1, 5)
    avg1[1], std1[1] = Avg_std_acc_d1(2, 5)
    avg1[2], std1[2] = Avg_std_acc_d1(4, 5)
    avg1[3], std1[3] = Avg_std_acc_d1(8, 5)
    avg1[4], std1[4] = Avg_std_acc_d1(None, 5)

    avg2[0], std2[0] = Avg_std_acc_d2(1, 5)
    avg2[1], std2[1] = Avg_std_acc_d2(2, 5)
    avg2[2], std2[2] = Avg_std_acc_d2(4, 5)
    avg2[3], std2[3] = Avg_std_acc_d2(8, 5)
    avg2[4], std2[4] = Avg_std_acc_d2(None, 5)

    print(avg1[3])

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    # Plot avg
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, avg1, width, label='Dataset 1')
    rects2 = ax.bar(x + width/2, avg2, width, label='Dataset 2')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Average accuracy')
    #ax.set_title('Average accuracy by tree depth over 5 generations')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    fig.tight_layout()
    plt.show()

    # Plot std
    fig_b, ax_b = plt.subplots()
    rects1_b = ax_b.bar(x - width/2, std1, width, label='Dataset 1')
    rects2_b = ax_b.bar(x + width/2, std2, width, label='Dataset 2')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax_b.set_ylabel('Standard deviation')
    #ax_b.set_title('Standard deviation by tree depth over 5 generations')
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(labels)
    ax_b.legend()
    fig_b.tight_layout()
    plt.show()
