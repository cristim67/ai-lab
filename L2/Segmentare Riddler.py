# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 13:05:05 2024

@author: Laborator
"""

import numpy as np
from skimage import io,color
import matplotlib.pyplot as plt

plt.close('all')


img=io.imread('biscuiti.jpg')
h,w=np.shape(img)
plt.figure(),plt.imshow(img,cmap="gray"),plt.colorbar(),plt.show()


def histograma(h,w,img):
    hist=np.zeros(256)
    for i in range(h):
        for j in range(w):
            hist[img[i,j]]+=1
            
    return hist/(h*w)

H=histograma(h,w,img)
plt.figure(),plt.plot(H),plt.show()

def prag_Riddler(Tcalc,H):
    L=256
    eps=0.0000000000001
    T=0
    while T!=Tcalc:
        T=Tcalc
        N1=0
        N2=0

        for i in range (T):
            N1=N1+i*H[i]
            N2=N2+H[i]
        miu0=N1/(N2+eps)
        
        N1=0
        N2=0
        for j in range (T,L):
            N1=N1+j*H[j]
            N2=N2+H[j]
        miu1=N1/(N2+eps)
        Tcalc=np.uint8((miu0+miu1)/2)
    return Tcalc


Tcalc=prag_Riddler(128,H)
SEGM=(img<=Tcalc)    #in ex nostru cu biscuiti avem mai mult fundal alb deci folosim <= sa facem fundalul negru si biscuitii albi
plt.figure(),plt.imshow(SEGM,cmap="gray"),plt.show()

            

