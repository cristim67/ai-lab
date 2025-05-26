# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 13:03:34 2024

@author: Laborator
"""


import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster

plt.close('all')


# verificare pe niste date 2D oarecare generate aleator
X1 = np.random.rand( 20 , 2 ) * 4 + 1
X2 = np.random.rand( 20 , 2 ) * 4 + 5
X3 = np.random.rand( 20 , 2 ) * 4 + 3

data = np.concatenate( (X1,X2,X3), axis = 0)
print(np.shape(data))

plt.figure(), plt.plot(data[:,0], data[:,1],'*'), plt.show()

kmeans = cluster.KMeans(n_clusters=3, random_state = None, n_init = 1).fit(data)
print(kmeans.labels_)
print(kmeans.cluster_centers_)
print(kmeans.inertia_)

dims = np.shape(data)
plt.figure()
for i in range(dims[0]):
    if(kmeans.labels_[i]==0):
        plt.plot(data[i,0],data[i,1],'r*')       
    elif(kmeans.labels_[i]==1):
        plt.plot(data[i,0],data[i,1],'g*')
    else:
        plt.plot(data[i,0],data[i,1],'b*')
plt.plot(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],'ok'),plt.show()
