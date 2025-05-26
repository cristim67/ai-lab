# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 13:57:29 2024

@author: Laborator
"""


import numpy as np
from skimage import io, measure
import matplotlib.pyplot as plt
import skfuzzy as fuzz

plt.close('all')

#img = io.imread('cuburi-mic.jpg')
img = io.imread('peppers-mic.png')
plt.figure(), plt.imshow(img), plt.show()
dims = np.shape(img)
print(dims)
H = dims[0]
W = dims[1]

R = img[:,:,0]
G = img[:,:,1]
B = img[:,:,2]
r = np.ndarray.flatten(R[:])
g = np.ndarray.flatten(G[:])
b = np.ndarray.flatten(B[:])
data = np.transpose(np.array([r,g,b]))
print(np.shape(data))

K = 5

centroizi, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(np.transpose(data), K, m = 2, error=0.005, maxiter = 1000, init=None)

print(centroizi)

# vizualizarea functiei obiectiv
plt.figure(), plt.plot(jm), plt.show()

# vizualizarea gradelor de apartenenta la fiecare clasa
for i in range(K):
    plt.figure(),plt.imshow(np.reshape(u[i,:], (H,W)),cmap="gray",interpolation='none'),plt.colorbar(),plt.show()

# defuzificare prin considerarea maximului
plt.figure(),plt.imshow(np.reshape(np.argmax(u, axis = 0), (H,W)),cmap="jet",interpolation='none'),plt.colorbar(),plt.show()


#marcam pixelii inconsistenti cu ultima eticheta
et=np.reshape(np.argmax(u, axis = 0), (H,W))
maxim=np.reshape(np.max(u, axis = 0), (H,W))
et[maxim<0.5]=K

plt.figure(),plt.imshow(et,cmap="jet",interpolation='none'),plt.colorbar(),plt.show()