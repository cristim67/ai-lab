# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 13:56:33 2024

@author: Laborator
"""


import numpy as np
from skimage import io
import matplotlib.pyplot as plt
from scipy import signal

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

Yzg=Y+np.random.normal(0,12,[Lines,Columns])
for i in range(0,Lines):
    for j in range(0, Columns):
        if (Yzg[i,j]>L-1):
            Yzg[i,j]=L-1
        else:
            if (Yzg[i,j]<0):
                Yzg[i,j]=0

Yzg=np.uint8(Yzg)

plt.figure(),plt.imshow(Y,cmap='gray',vmin=0,vmax=255),plt.colorbar(),plt.show()

plt.figure(),plt.imshow(Yzg,cmap='gray'),plt.colorbar(),plt.show()
fzx=signal.convolve2d(Yzg,np.array([[0,-1,0],[0,0,0],[0,1,0]]),boundary='symm', mode='same')
fzy=signal.convolve2d(Yzg,np.array([[0,0,0],[-1,0,1],[0,0,0]]),boundary='symm', mode='same')
gradYzg=np.abs(fzx)+np.abs(fzy)


def my_precrec(MASK, SEGM):
    epsilon=0.00001
    TP=SEGM*MASK
    FP=SEGM*(1-MASK)
    FN=(1-SEGM)*MASK
    prec=np.sum(TP)/(np.sum(TP)+np.sum(FP)+epsilon)
    rec=np.sum(TP)/(np.sum(TP)+np.sum(FN)+epsilon)
    return TP, FP, FN,prec, rec

max_gradYzg=np.max(gradYzg)

lista_prec=np.zeros(max_gradYzg)
lista_rec=np.zeros(max_gradYzg)
for T in range (0,max_gradYzg):
    SEGM=gradYzg>T
    TP, FP, FN, prec, rec = my_precrec(MASK, SEGM)
    lista_prec[T]=prec
    lista_rec[T]=rec
    # print(TP)
    # print(FP)
    # print(FN)
    
# print(lista_prec)
# print(lista_rec)

plt.figure(),plt.scatter(range(max_gradYzg),lista_prec),plt.scatter(range(max_gradYzg),lista_rec), plt.show()


dist_min=1
Tideal=0
for T in range (0,max_gradYzg):
    dist=abs(lista_prec[T]-lista_rec[T])
    if (dist<dist_min):
        dist_min=dist
        Tideal=T
        
print(Tideal)
plt.figure(),plt.imshow(gradYzg>Tideal,cmap='gray'),plt.colorbar(),plt.show()
        

    
