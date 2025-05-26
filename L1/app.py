# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 13:06:53 2024

@author: Laborator
"""


import matplotlib.pyplot as plt
import numpy as np
from skimage import color, io

plt.close('all')


img1 = io.imread('boabe-piper.jpg')
print(type(img1))
print(img1.dtype)
print(np.shape(img1))

plt.figure(),plt.imshow(img1),plt.colorbar(),plt.show()

img2 = io.imread('biscuiti.jpg')
print(' ')
print(type(img2))
print(img2.dtype)
print(np.shape(img2))
h2,w2=np.shape(img2)
plt.figure(),plt.imshow(img2,cmap="gray"),plt.colorbar(),plt.show()


img3 = io.imread('pills.jpg')
print(' ')
print(type(img3))
print(img3.dtype)
print(np.shape(img3))
h3,w3=np.shape(img3[:,:,1])
plt.figure(),plt.imshow(img3,cmap="gray"),plt.colorbar(),plt.show()

######################### 1)Histograma img si img cu zgomot 
#####histograma

def histograma(h,w,img):
    hist=np.zeros(256)
    for i in range(h):
        for j in range(w):
            hist[img[i,j]]+=1
            
    return hist/(h*w)

#histograma pentru img1 (color)
img1=color.rgb2gray(img1)
img1=np.uint8(img1*255)
h1,w1=np.shape(img1)
H1=histograma(h1,w1,img1)
plt.figure(),plt.plot(H1),plt.show()

#histograma pentru img2
H2=histograma(h2,w2,img2)
plt.figure(),plt.plot(H2),plt.show()

#histograma pentru img3  (alegem doar doua planuri din cele 3)
H3=histograma(h3,w3,img3[:,:,1])
plt.figure(),plt.plot(H3),plt.show()


###### zgomot gaussian
medie=0
disp=10
N1=np.random.normal(medie,disp,(h1,w1))
N2=np.random.normal(medie,disp,(h2,w2))
N3=np.random.normal(medie,disp,(h3,w3))

img2_zg=img2+N2
img2_zg=np.uint8(img2_zg)
h2_zg,w2_zg=np.shape(img2_zg)
H2_zg=histograma(h2_zg,w2_zg,img2_zg)
plt.figure(),plt.plot(H2_zg),plt.show()
plt.figure(),plt.imshow(img2_zg,cmap="gray"),plt.colorbar(),plt.show()




######################### 2)Segmentarea pe histograma: binarizarea
###pentru pastile img3
T1=150
T2=200
img3_plan=img3[:,:,1]
out=np.uint8(np.zeros([h3,w3]))
for i in range(0,h3):
    for j in range(0,w3):
        
        if (img3_plan[i,j]<T1):
            out[i,j]=0
        elif (T1<=img3_plan[i,j]<=T2):
            out[i,j]=1
        elif (img3_plan[i,j]>T2):
            out[i,j]=2

print(out.dtype)
plt.figure(),plt.imshow(out,cmap="gray"),plt.colorbar(),plt.show()