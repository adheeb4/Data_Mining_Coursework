# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 23:52:08 2023

@author: adheeb
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import OneClassSVM


def svm_plot():
    """This function plots the scatter plot using svm data"""
    plt.figure(1)
    plt.scatter(filtered_data["W0"], filtered_data["W2"])
    plt.scatter(outliers_svm["W0"], outliers_svm["W2"], c="r")
    plt.title('Week 0 Vs Week 1')
    plt.xlabel('Week 0')
    plt.ylabel('Week 1')
    plt.legend(labels=["Normal Data", "Outliers"])
    plt.show(1)


def k_dist():
    """This function plots the line graph of K-distance data"""
    plt.figure(figsize=(20, 10))
    plt.plot(distances)
    plt.title('K-Distance Graph', fontsize=30)
    plt.xlabel('Data Points sorted by distance', fontsize=20)
    plt.ylabel('Epsilon', fontsize=30)
    plt.show(2)


def dbscan_plot():
    """This function plots the scatter plot using dbscan data"""
    plt.figure(3)
    plt.scatter(data["W0"], data["W2"], c=colors)
    plt.title('Week 0 Vs Week 1')
    plt.xlabel('Week 0')
    plt.ylabel('Week 1')
    plt.show(3)


def neighbours():
    """This function finds the distances of different instances,
    sorts it and return the sorted data as an array"""
    neighbours = NearestNeighbors(n_neighbors=2)
    nbrs = neighbours.fit(dataframe[["W0", "W2"]])
    distances, indices = nbrs.kneighbors(dataframe[["W0", "W2"]])
    distances = np.sort(distances, axis=0)
    distances = distances[:, 1]
    return distances


# reading csv file using pandas
dataframe = pd.read_csv("Sales_Transactions_Dataset_Weekly.csv")
# c reating a dataframe using only two columns of the dataframe
filtered_data = dataframe[["W0", "W2"]]
# fitting data to SVM Model
model1 = OneClassSVM(kernel="rbf", gamma=0.001, nu=0.03).fit(filtered_data)
# creating an array of predicted values
y_pred = model1.predict(filtered_data)
# converting the array to list
y_pred_list = y_pred.tolist()
# creating blank array
outliers_index = []
# creating array of index of outliers
for i in np.arange(len(y_pred_list)):
    if y_pred_list[i] == -1:
        outliers_index.append(i)
# creating the an array of outliers
outliers_svm = filtered_data.iloc[outliers_index]
# calling the svm_plot() function
svm_plot()
# calling neighbours function
distances = neighbours()
# calling k_dist() function
k_dist()
# filtering datas to use two rows of dataframe
data = dataframe[["W0", "W2"]]
# fitting data to DBSCAn
model = DBSCAN(eps=10, min_samples=60).fit(data)
colors = model.labels_
# creating an array of outliers
outliers = data[model.labels_ == -1]
# calling dbscan_plot() function
dbscan_plot()
