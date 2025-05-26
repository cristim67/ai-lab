
# -*- coding: utf-8 -*-
"""
Created on Wed May 21 09:07:38 2025

@author: Tudor Arabagiu
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

#from fcmeans import FCM
import skfuzzy as fuzz
from numpy import ndarray as npnd
from scipy import signal
from skimage import color, feature, io, measure
from skimage.transform import resize
from sklearn import cluster, datasets

#1)

img = io.imread('pantaloni.jpg')

plt.figure(),plt.imshow(img,cmap="gray"),plt.title("Pantalon"),plt.colorbar(),plt.show()

print(np.shape(img))

def histograma(h,w,img):
    hist=np.zeros(256)
    for i in range(h):
        for j in range(w):
            hist[img[i,j]]+=1    
    return hist/(h*w)

h,w=np.shape(img)
hist= histograma(h,w,img)
plt.figure(),plt.plot(hist),plt.title("Histograma Pantalon"),plt.show()

#alegerea pragurilor (ne uitam pe histograma)
T1 = 100
T2 = 200


out = np.zeros((h, w), dtype=np.uint8)

for i in range(h):
    for j in range(w):
        if (img[i,j]<T1):
            out[i,j]=2                  
        elif (T1<=img[i,j]<=T2):
            out[i,j]=0                  
        elif (img[i,j]>T2):
            out[i,j]=1

plt.figure(), plt.imshow(out, cmap='gray')
plt.title("Segmentare pantalon"), plt.colorbar(), plt.show()

#2)

Y = io.imread('cercuri-dreptunghiuri.jpg')

if Y.ndim == 3:
    Y = color.rgb2gray(Y)

fx=signal.convolve2d(Y,np.array([[0,-1,0],[0,0,0],[0,1,0]]),boundary='symm', mode='same')
fy=signal.convolve2d(Y,np.array([[0,0,0],[-1,0,1],[0,0,0]]),boundary='symm', mode='same')
gradY=np.abs(fx)+np.abs(fy)
     
plt.figure(),plt.imshow(gradY,cmap='gray'),plt.colorbar(),plt.show()

# 3. Binarizează gradientul ca să obții doar contururile
thresh = np.mean(gradY) * 1.5  # prag adaptiv
contururi = gradY > thresh

# 4. Afișează imaginea cu contururi
plt.figure(), plt.imshow(contururi, cmap='gray')
plt.title("Contururile tuturor obiectelor"), plt.show()

# 5. Calculează procentul de pixeli de contur
procent = 100 * np.sum(contururi) / contururi.size
print("Procentul ocupat de pixeli de contur",procent,"%")

#3)
# Etichetare și filtrare după arie și perimetru
LabelImage, nums = measure.label(contururi, return_num=True)
ALLPROPS = measure.regionprops(LabelImage)

# data = np.zeros((nums,2))
# for i in range(nums):
#     dummy = ALLPROPS[i]
#     data[i,0] = dummy.area
#     data[i,1] = dummy.perimeter
    
# print(data)

# Parametri pentru eliminare
prag_arie = 65
prag_perimetru = 55

# Filtrare obiecte mici
masca_filtrata = np.zeros_like(LabelImage)
for i, prop in enumerate(ALLPROPS):
    if prop.area >= prag_arie and prop.perimeter >= prag_perimetru:
        masca_filtrata[LabelImage == (i + 1)] = 1
        
plt.figure(), plt.imshow(masca_filtrata, cmap='gray'),plt.title("Imagine după eliminarea obiectelor mici (arie + perimetru)"), plt.colorbar(), plt.show()

# PAS 3: Etichetare nouă după filtrare
LabelImage2, nums2 = measure.label(masca_filtrata, return_num=True)
PROPS2 = measure.regionprops(LabelImage2)

# Selectarea obiectului cel mai îndepărtat de centrul imaginii (doar partea superioară)
H, W = LabelImage2.shape
centru = np.array([H // 2, W // 2])

distant_max = -1
index_final = -1

for i, prop in enumerate(PROPS2):
    y, x = prop.centroid
    if y < H // 2:  # doar partea superioară
        dist = np.linalg.norm(np.array([y, x]) - centru) #distanța euclidiană dintre două puncte 2D, np.linalg.norm() = norma L2 (distanța euclidiană)
        if dist > distant_max:
            distant_max = dist
            index_final = i + 1

# Creare mască finală cu obiectul selectat
masca_finala = (LabelImage2 == index_final)

# Afișare rezultat
plt.figure(), plt.imshow(masca_finala, cmap="gray"), plt.title("Obiectul cel mai îndepărtat de centru (partea superioară)"), plt.colorbar(), plt.show()

#4)
# Citire imagine și pregătire
image = io.imread('textura1.jpg')  
if image.ndim == 3:
    image = color.rgb2gray(image)

L = 256
d = 16
image = np.uint8(image / d)

dims = image.shape

# Filtre de orientare
f0 = signal.convolve2d(image, [[-1, -1], [1, 1]], mode='same')
f90 = signal.convolve2d(image, [[-1, 1], [-1, 1]], mode='same')
f45 = signal.convolve2d(image, [[1.41, 0], [0, -1.41]], mode='same')
f135 = signal.convolve2d(image, [[0, 1.41], [-1.41, 0]], mode='same')
funif = signal.convolve2d(image, [[2, -2], [-2, 2]], mode='same')

# Alege cele 2 orientări cele mai distincte în histogramă
# Analizăm orientările
img_orient = np.zeros(dims)
for i in range(dims[0]):
    for j in range(dims[1]):
        responses = np.abs([f0[i,j], f45[i,j], f90[i,j], f135[i,j], funif[i,j]])
        img_orient[i,j] = np.argmax(responses)

# Afișăm orientările dominante
plt.figure(), plt.imshow(img_orient, cmap='jet'), plt.colorbar()
plt.title("Orientare dominantă per pixel"), plt.show()

hist_orient = np.histogram(img_orient, bins=5)[0]
print("Histogramă orientări:", hist_orient)

# Selectăm cele mai frecvente 2 orientări
top2_indices = np.argsort(hist_orient)[-2:]
print("Cele mai relevante orientări:", top2_indices)

# Parametrii pentru blocuri
size_patch = 64
step = size_patch // 2
nr_patch = dims[0] // step - 1

features = np.zeros((nr_patch * nr_patch, 2))
index = 0

for i in range(nr_patch):
    for j in range(nr_patch):
        patch_orient = img_orient[i*step:(i+2)*step, j*step:(j+2)*step]
        h = np.histogram(patch_orient, bins=5)[0]

        # Folosim doar cele mai relevante 2 direcții
        features[index, 0] = h[top2_indices[0]]
        features[index, 1] = h[top2_indices[1]]
        index += 1

# Clustering pe cele 2 dimensiuni alese
kmeans = cluster.KMeans(n_clusters=3, random_state=0).fit(features)
etichete = kmeans.labels_

# Afișare segmentare finală
plt.figure(), plt.imshow(etichete.reshape((nr_patch, nr_patch)), cmap='jet'),plt.title("Segmentare finală cu 3 texturi (pe baza histogramă orientare)"), plt.colorbar(), plt.show()

