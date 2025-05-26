# -*- coding: utf-8 -*-
"""
Created on Mon May 20 23:25:12 2024

@author: ioana
"""

########### Bilet S4 #############

## CRED CA ASA ERA  CERINTA 

# -> 1) Formati o img de dim 180x180 cu 3 bare orizontale de dim egale a caror intensitati de gri sunt in ordine de sus in jos: 20,100,200. 
#       Adaugati zgomot de dispersie 18 si alegeti pragurile de segmentare (manuala) corespunzatoare a.i. bara de sus sa aiba eticheta 1,
#       bara din mijloc sa aiba eticheta neagra , bara de sus eticheta alba si bara de jos eticheta gri. Afisati imaginea rezultata.
# -> 2) Pt img segmentata la 1) afisati procentul de pixeli asociat barii situate cel mai jos detectati gresit.
# -> 3) Pt img S3.png folositi convenabil perimetrul si aria a.i. sa delimitati cercurile de restul formelor cu ajutorul unui algoritm de clustering.
#       Afisati rezultatul final
# -> 4) Det automat si afisati conturul obiectului cel mai aproape de un "cerc ideal".

import numpy as np
from skimage import io, color, measure
import matplotlib.pyplot as plt
from scipy import signal
from sklearn import cluster

plt.close('all')


def add_gaussian_noise(img,disp,L):
    # parametrii : imaginea originala, dispersia zgomotului
    h,w = img.shape
    noise = np.random.normal(0,disp,(h,w))
    img_noise = np.zeros([h,w])
    img_noise = img + noise
    for i in range(0, h):
        for j in range(0, w):
            if (img_noise[i,j]>L-1):
                img_noise[i,j]=L-1
            else:
                if (img_noise[i,j]<0):
                    img_noise[i,j]=0
    img_noise = np.uint8(img_noise)
    # intoarce imaginea cu zgomot
    return img_noise


def histograma(h,w,img):
    hist=np.zeros(256)
    for i in range(h):
        for j in range(w):
            hist[img[i,j]]+=1    
    return hist/(h*w)

################################################ EX 1
L=256
disp=18
img1 = np.zeros([180, 180], dtype='uint8')
img1[:60, :] = 20
img1[60:120, :] = 100
img1[120:, :] = 200

##afisare img originala  
plt.figure("Imagine initiala"), plt.imshow(img1, cmap='gray', vmin=0, vmax=255), plt.colorbar(), plt.title("Imagine initiala"), plt.show()


##adaugarea de zgomot si afisare img zgomot
img1_zg = add_gaussian_noise(img1, disp, L)
plt.figure("Imagine cu zgomot"), plt.imshow(img1_zg, cmap='gray', vmin=0, vmax=255), plt.colorbar(), plt.title("Imagine cu zgomot"),plt.show()


#calcul histograma pt alegerea pragurilor
h1,w1=np.shape(img1_zg)
hist=histograma(h1,w1,img1_zg)
plt.figure("Histograma img1_zg"),plt.plot(hist),plt.title("Histograma img1_zg"),plt.show()

#segmentare manuala
#alegem pragurile (ne uitam pe histograma pt a le alege)
T1=50
T2=150
img1_segm=np.uint8(np.zeros([h1,w1]))
for i in range(0,h1):
    for j in range(0,w1):
        
        if (img1_zg[i,j]<T1):
            img1_segm[i,j]=2                  #bara de sus eticheta 1
        elif (T1<=img1_zg[i,j]<=T2):
            img1_segm[i,j]=0                 
        elif (img1_zg[i,j]>T2):
            img1_segm[i,j]=1

print(img1_segm.dtype)
plt.figure(),plt.imshow(img1_segm,cmap="gray"), plt.title(" Img1 segmentata"),plt.colorbar(),plt.show()



################################################ EX 2
pixeli_bara_jos = 60 * 180        #fiecare bara are 70 de pixeli latime si 210 lungime
#bara de jos are eticheta 0 si de aceea se scrie !=0
pixeli_gresiti = np.sum(img1_segm[120:, :] != 1)   #bara de jos e de la 140-210 pixeli
print(pixeli_gresiti / pixeli_bara_jos * 100, '%')


################################################ EX 3

img1 = io.imread("subiect3.png")
print(np.shape(img1))

# de verificat sa nu cumva sa fie imaginea color CMYK => o fac uint8 gri
# daca imaginea este gri => o facem uint8

img1 = np.uint8(255*color.rgb2gray(img1))

plt.figure(), plt.imshow(img1,cmap='gray',vmin=0,vmax=255),plt.colorbar(),plt.show()


h,w=np.shape(img1)
hist=histograma(h,w,img1)
plt.figure("Histograma img1"),plt.plot(hist),plt.title("Histograma img1"),plt.show()

BW=np.uint8(img1>50) #imaginea binarizata
plt.figure(),plt.imshow(BW,cmap="gray"),plt.colorbar(),plt.show()

[LabelImage, nums]=measure.label(BW,return_num='True')
print(nums)
#plt.figure(figsize=(10,20)),plt.imshow(LabelImage,cmap="jet"),plt.colorbar(),plt.show()
#plt.figure(figsize=(10,20)),plt.imshow(LabelImage,cmap="jet",interpolation='none'),plt.colorbar(),plt.show()

# Calcularea proprietăților regiunilor
ALLPROPS = measure.regionprops(LabelImage)
size=len(ALLPROPS)
print(size)
features = np.zeros([nums,1]) ### de modificat nr obiecte ###
for i in range(nums):
        features[i,0] = (ALLPROPS[i].perimeter**2)/(ALLPROPS[i].area*4*np.pi)

# clustering
kmeans = cluster.KMeans(n_clusters=2, random_state=0).fit(features)
etichete = kmeans.labels_
centroizi = kmeans.cluster_centers_
img_cluster = np.zeros(np.shape(LabelImage))
h,w = np.shape(LabelImage)
for i in range(0,h):
    for j in range(0,w):
        if LabelImage[i, j] != 0:
            img_cluster[i,j]=etichete[LabelImage[i,j]-1]+1
            
plt.figure(), plt.imshow(img_cluster, cmap="jet",interpolation='none'), plt.colorbar(), plt.show()


################################################ EX 4
#pt un cerc perfect, masura de compactitate este 1, pt celelalte forme e >1

#pp ca primul obiect are masura de compaxtitate cea mai mica
minim = (ALLPROPS[0].perimeter**2)/(ALLPROPS[0].area*4*np.pi)
obiect = 0  #in variabila obiect stocam eticheta obiectului cautat
for i in range(nums):
    rap =  (ALLPROPS[i].perimeter**2)/(ALLPROPS[i].area*4*np.pi)
    if(rap < minim):
        minim = rap
        obiect =i
        
img_obiect = np.zeros(np.shape(LabelImage))
img_obiect = np.uint8(LabelImage==obiect+1)

dims=np.shape(img_obiect)

print(dims)

Lines=dims[0]
Columns=dims[1]

Y=img_obiect[0:Lines,0:Columns]
plt.figure(),plt.imshow(Y,cmap='gray'),plt.colorbar(), plt.title("Img originala"),plt.show()


#bordare pe ultima linie si coloana
YY=np.zeros([Lines+1, Columns+1])
YY[0:Lines,0:Columns]=Y               #pune in YY img initiala
YY[Lines,0:Columns]=Y[Lines-1,:]      #copiaza ultima linie
YY[0:Lines,Columns]=Y[:,Columns-1]    #copiaza ultima coloana
YY[Lines,Columns]=Y[Lines-1,Columns-1]   #adauga linia si coloana copiate anterior
print(np.shape(YY))  #s-a marit cu o linie si o coloana

plt.figure(),plt.imshow(YY,cmap='gray'),plt.colorbar(), plt.title("Img bordata"),plt.show()

#derivatele pe orizontala si verticala:
fx=np.zeros([Lines, Columns])
fy=np.zeros([Lines, Columns])

for l in range(0,Lines):
    for c in range(0,Columns):
        fx[l,c]=YY[l,c]-YY[l+1,c]
        
for l in range(0,Lines):
    for c in range(0,Columns):        
        fy[l,c]=YY[l,c]-YY[l,c+1]


gradY=np.abs(fx)+np.abs(fy)
plt.figure(),plt.imshow(gradY,cmap='gray'),plt.colorbar(),plt.show() #evidentiaza tot conturul