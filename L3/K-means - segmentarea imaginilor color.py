# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 13:18:55 2024

@author: Laborator
"""


import numpy as np
from skimage import io, measure
import matplotlib.pyplot as plt
from sklearn import cluster,datasets
import matplotlib as mpl


plt.close('all')

# citim o imagine color
#img=io.imread('dipum.jpg')
#img=io.imread('cuburi-mic.jpg')
img=io.imread('mm-mic.jpg')


plt.figure(),plt.imshow(img),plt.show()
dims = np.shape(img)
print(dims)
# luam separat planurile de culoare
R = img[:,:,0]
G = img[:,:,1]
B = img[:,:,2]
# le vectorizam 
r = np.ndarray.flatten(R[:])
print(np.shape(r))
g=np.ndarray.flatten(G[:])
b=np.ndarray.flatten(B[:])

# ne formam setul de date
data = np.transpose(np.array([r,g,b]))
print(np.shape(data))


K = 8
kmeans = cluster.KMeans(n_clusters=K, random_state=0).fit(data)
etichete = kmeans.labels_
centroizi = kmeans.cluster_centers_

print(centroizi)
print(etichete)

harta=np.array(centroizi/255)
harta=mpl.colors.ListedColormap(harta)

plt.figure(), plt.imshow(img), plt.show()
plt.figure(), plt.imshow(np.reshape(etichete, (dims[0],dims[1])), cmap=harta,interpolation='none'), plt.colorbar(), plt.show()

print(np.shape(etichete))

for i in range(K):
    plt.figure(),plt.imshow(np.uint8(np.reshape(etichete, (dims[0],dims[1]))==i),cmap="gray",interpolation='none'),plt.colorbar(),plt.show()
