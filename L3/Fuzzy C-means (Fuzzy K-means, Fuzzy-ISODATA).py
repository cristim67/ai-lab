# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 13:56:39 2024

@author: Laborator
"""


import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz

plt.close('all')

X1 = np.random.rand( 20 , 2 ) * 3 + 1
X2 = np.random.rand( 20 , 2 ) * 3 + 5
X3 = np.random.rand( 20 , 2 ) * 3 + 3
data = np.concatenate( (X1,X2,X3), axis = 0)
print(np.shape(data))

plt.figure(), plt.plot(data[:,0],data[:,1],'*'), plt.show()


centroizi, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(np.transpose(data), 3, m = 2, error=0.005, maxiter = 1000, init=None)

print(centroizi)
print(u)
print(np.shape(u))

plt.figure(), plt.plot(data[:,0], data[:,1],'*'), 
plt.plot(centroizi[:,0],centroizi[:,1],'ok')
plt.show()

print(np.shape(u))

#print(np.sum(u))
#print(np.sum(u,axis=0))

print(u[:,4])

dims = np.shape(data)

plt.figure()
for i in range(dims[0]):
    if(np.argmax(u[:,i])==0):
        plt.plot(data[i,0], data[i,1],'r*')       
    elif(np.argmax(u[:,i])==1):
        plt.plot(data[i,0], data[i,1],'g*')
    else:
        plt.plot(data[i,0], data[i,1],'b*')
plt.plot(centroizi[:,0],centroizi[:,1],'ok')
plt.show()

