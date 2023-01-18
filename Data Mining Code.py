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

dataframe = pd.read_csv("Sales_Transactions_Dataset_Weekly.csv")

neighbours = NearestNeighbors(n_neighbors=2)
nbrs = neighbours.fit(dataframe[["W0", "W2"]])
distances, indices = nbrs.kneighbors(dataframe[["W0", "W2"]])


distances = np.sort(distances, axis=0)
distances = distances[:,1]
plt.figure(figsize=(20,10))
plt.plot(distances)
plt.title('K-Distance Graph',fontsize=30)
plt.xlabel('Data Points sorted by distance',fontsize=20)
plt.ylabel('Epsilon',fontsize=30)
plt.show()


data = dataframe[["W0", "W2"]]
model = DBSCAN(eps = 10, min_samples = 60).fit(data)
colors = model.labels_
plt.figure(1)
plt.scatter(data["W0"], data["W2"], c = colors)
plt.title('Week 0 Vs Week 1')
plt.xlabel('Week 0')
plt.ylabel('Week 1')
plt.show(1)

outliers = data[model.labels_ == -1]


filtered_data = dataframe[["W0","W2"]]
model1 = OneClassSVM(kernel = "rbf", gamma =0.001, nu = 0.03).fit(filtered_data)
y_pred = model1.predict(filtered_data)
y_pred_list = y_pred.tolist()

index = []

for i in np.arange(len(y_pred_list)):
    if y_pred_list[i] == -1:
        index.append(i)
        
values = filtered_data.iloc[index]
plt.figure(2)
plt.scatter(filtered_data["W0"], filtered_data["W2"])
plt.scatter(values["W0"], values["W2"], c = "r")
plt.title('Week 0 Vs Week 1')
plt.xlabel('Week 0')
plt.ylabel('Week 1')
plt.legend(labels=["Normal Data","Outliers",])
plt.show(2)