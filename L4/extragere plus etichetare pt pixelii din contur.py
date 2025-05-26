# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 14:35:54 2024

@author: Laborator
"""


import numpy as np
from skimage import io, measure, color
import matplotlib.pyplot as plt
from scipy import signal

plt.close('all')

L=256


img=io.imread('forme.jpg')
dims=np.shape(img)

img=img[:,:,1]
dims=np.shape(img)

print(dims)

Lines=dims[0]
Columns=dims[1]

Y=img[0:Lines,0:Columns]
fzx=signal.convolve2d(Y,np.array([[0,-1,0],[0,0,0],[0,1,0]]),boundary='symm', mode='same')
fzy=signal.convolve2d(Y,np.array([[0,0,0],[-1,0,1],[0,0,0]]),boundary='symm', mode='same')
gradY=np.abs(fzx)+np.abs(fzy)
plt.figure(),plt.imshow(gradY,cmap='gray'),plt.colorbar(),plt.show()

gradY=gradY>100
plt.figure(),plt.imshow(gradY,cmap='gray'),plt.show()
[LabelImage,nums]=measure.label(gradY,return_num='True')
print(nums)
plt.figure(),plt.imshow(LabelImage,cmap="jet"),plt.colorbar(),plt.show()




##### determini conturul cu perimetrul/ aria maxim/maxima