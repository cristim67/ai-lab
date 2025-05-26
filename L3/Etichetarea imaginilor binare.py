# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 14:15:47 2024

@author: Laborator
"""


import numpy as np
from skimage import io, measure
import matplotlib.pyplot as plt
from sklearn import cluster
import matplotlib as mpl

plt.close('all')


#img=io.imread('biscuiti.jpg')
Y=io.imread('test-obiecte.bmp')
dims=np.shape(Y)
print(dims)

H=dims[0]
W=dims[1]
plt.figure(),plt.imshow(Y,cmap="gray"),plt.colorbar(),plt.show()

h=np.histogram(Y,bins=256,range=(0,256),density=True)
plt.figure(),plt.plot(h[0]),plt.show()
BW=np.uint8(Y>100)
plt.figure(),plt.imshow(BW,cmap="gray"),plt.colorbar(),plt.show()

[LabelImage, nums]=measure.label(BW,return_num='True')
print(nums)
plt.figure(figsize=(10,20)),plt.imshow(LabelImage,cmap="jet"),plt.colorbar(),plt.show()
plt.figure(figsize=(10,20)),plt.imshow(LabelImage,cmap="jet",interpolation='none'),plt.colorbar(),plt.show()

# print(nums)
# print(np.min(LabelImage))
# print(np.max(LabelImage))

# hL=np.zeros(nums+1)
# for i in range (0,H):
#     for j in range (0,W):
#         hL[LabelImage[i,j]]+=1
        
# plt.figure(),plt.plot(hL),plt.show()

# plt.figure(),plt.plot(hL[1:nums]),plt.show()
# print(hL[0:nums])

# componenta_extrasa=41
# COMP=np.uint8(LabelImage==componenta_extrasa)
# plt.figure(),plt.imshow(COMP,cmap="gray"),plt.colorbar(),plt.show()

ALLPROPS=measure.regionprops(LabelImage)
#indexul obiectelor din imagine pentru structura de proprietati este  ETICHETA-1
dummy=ALLPROPS[14]
print(dummy.centroid)
print(dummy.area)
print(dummy.orientation)
print(dummy.perimeter)
print(dummy.moments_hu)

arie=np.zeros([nums,1],dtype='uint8')
for i in range(nums):
    arie[i]=ALLPROPS[i].area
    perimetru[i]=ALLPROPS[i].perimeter
    
    
h,w=np.shape(LabelImage)
new_img=np.zeros(h,w)

dims = np.shape(Y)

K = 8
kmeans = cluster.KMeans(n_clusters=K, random_state=0).fit(data)
etichete = kmeans.labels_
centroizi = kmeans.cluster_centers_

print(centroizi)
print(etichete)

# harta=np.array(centroizi/255)
# harta=mpl.colors.ListedColormap(harta)

# plt.figure(), plt.imshow(Y), plt.show()
# plt.figure(), plt.imshow(np.reshape(etichete, (dims[0],dims[1])), cmap=harta,interpolation='none'), plt.colorbar(), plt.show()

# print(np.shape(etichete))

# for i in range(K):
#     plt.figure(),plt.imshow(np.uint8(np.reshape(etichete, (dims[0],dims[1]))==i),cmap="gray",interpolation='none'),plt.colorbar(),plt.show()

for i in range (nums):
    new_img[LabelImage==i+1]=etichete[i]+1
    
    
plt.figure(), plt.imshow(new_img), plt.show()


