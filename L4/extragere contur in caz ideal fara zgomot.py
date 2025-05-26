# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 13:09:40 2024

@author: Laborator
"""
 
import numpy as np
from skimage import io
import matplotlib.pyplot as plt

plt.close('all')

L=256

img=io.imread('biscuiti.jpg')
dims=np.shape(img)

print(dims)

Lines=dims[0]
Columns=dims[1]

Y=img[0:Lines,0:Columns]

#Y=100*np.ones([Lines, Columns])
#Y[100:150,100:150]=200

#bordare pe ultima linie si coloana
YY=np.zeros([Lines+1, Columns+1])
YY[0:Lines,0:Columns]=Y
YY[Lines,0:Columns]=Y[Lines-1,:]
YY[0:Lines,Columns]=Y[:,Columns-1]
YY[Lines,Columns]=Y[Lines-1,Columns-1]
print(np.shape(YY))

plt.figure(),plt.imshow(YY,cmap='gray'),plt.colorbar(),plt.show()

fx=np.zeros([Lines, Columns])
fy=np.zeros([Lines, Columns])

for l in range(0,Lines):
    for c in range(0,Columns):
        fx[l,c]=YY[l,c]-YY[l+1,c]
        
for l in range(0,Lines):
    for c in range(0,Columns):        
        fy[l,c]=YY[l,c]-YY[l,c+1]


        
plt.figure(),plt.imshow(fx,cmap='gray'),plt.colorbar(),plt.show()
plt.figure(),plt.imshow(fy,cmap='gray'),plt.colorbar(),plt.show()

gradY=np.abs(fx)+np.abs(fy)
plt.figure(),plt.imshow(gradY,cmap='gray'),plt.colorbar(),plt.show()

# Harta binara de contururi
plt.figure(),plt.imshow(gradY>75,cmap='gray'),plt.colorbar(),plt.show()

# Harta binara de contururi
plt.figure(figsize=(8,16)),plt.imshow(gradY>80,cmap='gray'),plt.colorbar(),plt.show()