# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 13:24:55 2024

@author: Laborator
"""


import matplotlib.pyplot as plt
import numpy as np
from skimage import io

plt.close('all')

L=256

# Aici fac o imagine ideala

Y = 100*np.ones([200,200])
Y[:,100:200]=180

MASK = np.zeros([200,200])
MASK[:,99:101]=1

# aici adaug zgomot imaginii ideale de luminanta
dims=np.shape(Y)
Lines=dims[0]
Columns=dims[1]

Yzg=Y+np.random.normal(0,30,[Lines,Columns])
for i in range(0,Lines):
    for j in range(0, Columns):
        if (Yzg[i,j]>L-1):
            Yzg[i,j]=L-1
        else:
            if (Yzg[i,j]<0):
                Yzg[i,j]=0

Yzg=np.uint8(Yzg)

plt.figure(),plt.imshow(Y,cmap='gray',vmin=0,vmax=255),plt.colorbar(),plt.show()
from scipy import signal

fx=signal.convolve2d(Y,np.array([[0,-1,0],[0,0,0],[0,1,0]]),boundary='symm', mode='same')
fy=signal.convolve2d(Y,np.array([[0,0,0],[-1,0,1],[0,0,0]]),boundary='symm', mode='same')
gradY=np.abs(fx)+np.abs(fy)
     
plt.figure(),plt.imshow(gradY,cmap='gray'),plt.colorbar(),plt.show()
plt.figure(),plt.plot(gradY[100,:]),plt.show()

print(gradY[100,98:102])

plt.figure(),plt.imshow(Yzg,cmap='gray'),plt.colorbar(),plt.show()
fzx=signal.convolve2d(Yzg,np.array([[0,-1,0],[0,0,0],[0,1,0]]),boundary='symm', mode='same')
fzy=signal.convolve2d(Yzg,np.array([[0,0,0],[-1,0,1],[0,0,0]]),boundary='symm', mode='same')
gradYzg=np.abs(fzx)+np.abs(fzy)
     
plt.figure(),plt.imshow(gradYzg,cmap='gray'),plt.colorbar(),plt.show()
plt.figure(),plt.plot(gradYzg[100,:]),plt.show()
plt.figure(),plt.imshow(gradYzg>100,cmap='gray'),plt.colorbar(),plt.show()
print(gradYzg[100,98:102]>170)