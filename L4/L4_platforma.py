# -*- coding: utf-8 -*-
"""
Created on Sun May 19 17:37:39 2024

@author: Ema
"""

import numpy as np
from skimage import io
import matplotlib.pyplot as plt
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

import numpy as np
from skimage import io
import matplotlib.pyplot as plt
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

fzx=signal.convolve2d(Yzg,np.array([[-1,-1,-1],[0,0,0],[1,1,1]]),boundary='symm', mode='same')
fzy=signal.convolve2d(Yzg,np.array([[-1,0,1],[-1,0,1],[-1,0,1]]),boundary='symm', mode='same')
gradYzg=np.abs(fzx)+np.abs(fzy)
     
plt.figure(),plt.imshow(gradYzg,cmap='gray'),plt.colorbar(),plt.show()
plt.figure(),plt.plot(gradYzg[100,:]),plt.show()

plt.figure(),plt.imshow(gradYzg>300,cmap='gray'),plt.colorbar(),plt.show()

from numpy import ndarray as npnd
hhh=np.histogram(npnd.flatten(gradYzg),256,density=True)

histo=hhh[0]/np.sum(hhh[0])

h=hhh[1]

#plt.figure(),plt.plot(h[1:],hhh[0]),plt.show()
plt.figure(),plt.plot(h[1:],histo),plt.show()
#print(histo)

# voi alege drept contur 1% dintre punctele imaginii = segmentare pe histograma cumulativa
H=np.cumsum(histo)
#print(H)
plt.figure(),plt.plot(h[1:],H),plt.show()

prag=np.argmin(np.abs(H-0.99))
print(prag)
print(h[prag])
plt.figure(),plt.imshow(gradYzg>h[prag],cmap="gray"),plt.colorbar(),plt.show()
TEST = gradYzg>h[prag]
TEST = gradYzg>400


plt.figure(),plt.imshow(MASK,cmap="gray"),plt.colorbar(),plt.show()
plt.figure(),plt.imshow(TEST,cmap="gray"),plt.colorbar(),plt.show()
def my_precrec(MASK, SEGM):
    epsilon=0.00001
    TP=SEGM*MASK
    FP=SEGM*(1-MASK)
    FN=(1-SEGM)*MASK
    prec=np.sum(TP)/(np.sum(TP)+np.sum(FP)+epsilon)
    rec=np.sum(TP)/(np.sum(TP)+np.sum(FN)+epsilon)
    return prec, rec

prec, rec = my_precrec(MASK, TEST)
print(prec)
print(rec)