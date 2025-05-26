# -*- coding: utf-8 -*-
"""
Created on Mon May 20 20:43:56 2024

@author: ioana
"""

#1) Cititi si afisati img alts2.png. Adaugati zgomot aditiv gaussian de dispersie 13.
#Sa se segmenteze imaginea cu praguri alese manual, a.i. fundalul sa fie negru, 
#patratele si cercurile sa aiba acelasi ton de gri, iar dreptunghiurile sa fie albe. 
#Afisati imaginea segmentata.
#2) Folosind img masca.png si rez de la 1) det procentul de pixeli de "dreptunghi" identificati corect
#3)Etichetati img alts2.png. Folositi o metoda automata prin care sa ramaneti in imagine doar cu dreptunghiuri si patrate.
#Afisati intr-o figura noua rezultatul. 
#Etichetati din nou img rezultata a.i. sa nu existe etichete lipsa.
#Pt aceasta folositi convenabil aria si/sau perimetrul a.i. sa separati dreptunghiurile de patrate printr-un procedeu de clustering.
#Afisati rezultatul final
#4) Det automat si afisati conturul obiectului cu suprafata cea mai mare.


import numpy as np
from skimage import io, color, measure
import matplotlib.pyplot as plt
from scipy import signal
from sklearn import cluster

plt.close('all')


################################################ EX 1
img= io.imread('alts2.png')
print(np.shape(img))
img_plan=img[:,:,1]
plt.figure(),plt.imshow(img_plan,cmap="gray"),plt.title("Img originala"),plt.colorbar(),plt.show()


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


L=256
disp=13

##adaugarea de zgomot si afisare img zgomot
img_zg = add_gaussian_noise(img_plan, disp, L)
plt.figure("Imagine cu zgomot"), plt.imshow(img_zg, cmap='gray', vmin=0, vmax=255), plt.colorbar(), plt.title("Imagine cu zgomot"),plt.show()

def histograma(h,w,img):
    hist=np.zeros(256)
    for i in range(h):
        for j in range(w):
            hist[img[i,j]]+=1    
    return hist/(h*w)

#calcul histograma pt alegerea pragurilor
h,w=np.shape(img_zg)
hist=histograma(h,w,img_zg)
plt.figure("Histograma img_zg"),plt.plot(hist),plt.title("Histograma img_zg"),plt.show()

#segmentare manuala
#alegem pragurile (ne uitam pe histograma pt a le alege)
T1=100
T2=200
img_segm=np.uint8(np.zeros([h,w]))
for i in range(0,h):
    for j in range(0,w):
        
        if (img_zg[i,j]<T1):
            img_segm[i,j]=0                 #fundalul
        elif (T1<=img_zg[i,j]<=T2):
            img_segm[i,j]=1                  #patrate si cercuri
        elif (img_zg[i,j]>T2):
            img_segm[i,j]=2                 #dreptunghiuri

print(img_segm.dtype)
plt.figure(),plt.imshow(img_segm,cmap="gray"), plt.title(" Img segmentata"),plt.colorbar(),plt.show()

################################################ EX 2
mask= io.imread('masca.png')
mask_plan=mask[:,:,1]

def PrecRec(MASK,SEGM):#mask -> masca, segm -> segmentarea facuta
    epsilon=0.00001
    TP=SEGM*MASK
    FP=SEGM*(1-MASK)
    FN=(1-SEGM)*MASK
    prec=np.sum(TP)/(np.sum(TP)+np.sum(FP)+epsilon) #cat % s-a segmentat corect
    rec=np.sum(TP)/(np.sum(TP)+np.sum(FN)+epsilon) # cat % s-a segmentat din corect
    return TP


plt.figure(),plt.imshow(mask_plan,cmap='gray'),plt.colorbar(),plt.show()
plt.figure(),plt.imshow((mask_plan+2*img_segm),cmap='tab10',interpolation='none'),plt.colorbar(),plt.show()

#nu ne trebuie rec si prec,de asta am modificat functia la return
TP= PrecRec(mask_plan,img_segm)
# prec,rec = PrecRec(mask_plan,img_segm)
# print(prec)
# print(rec)
print(np.sum(TP)/np.sum(mask==255))


################################################ EX 3

def prag_Otsu(h): #h este histograma imaginii
    eps=0.0000000000001
    criteriu=np.zeros(256)
    L = 256
    for T in range (0,L):
        P0=0
        mu0=0
        for i in range(0,T):
            P0+=h[i]
            mu0+=i*h[i]
        mu0=mu0/(P0+eps)
    
        P1=0
        mu1=0
        for i in range(T,L):
            P1+=h[i]
            mu1+=i*h[i]
        mu1=mu1/(P1+eps)
        criteriu[T]=P0*mu0*mu0+P1*mu1*mu1
    THR = np.argmax(criteriu)
    return THR

#calcul histograma pt alegerea pragurilor
h1,w1=np.shape(img_plan)
hist1=histograma(h1,w1,img_plan)
plt.figure("Histograma img_zg"),plt.plot(hist1),plt.title("Histograma img_zg"),plt.show()

prag = prag_Otsu(hist1)
SEGM=(img_plan>prag)    #in ex nostru cu biscuiti avem mai mult fundal alb deci folosim <= sa facem fundalul negru si biscuitii albi
SEGM=np.uint8(SEGM)
plt.figure(),plt.imshow(SEGM,cmap='gray'),plt.title("Imagine biscuiti segmentata cu prag Otsu"),plt.show()

# ################### ema
# #################################### ->3)

# # Determinarea automata a pragului de segmentare cu metoda OTSU
# def prag_Otsu(h): #h este histograma imaginii
#     eps=0.0000000000001
#     criteriu=np.zeros(256)
#     L = 256
#     for T in range (0,L):
#         P0=0
#         mu0=0
#         for i in range(0,T):
#             P0+=h[i]
#             mu0+=i*h[i]
#         mu0=mu0/(P0+eps)
    
#         P1=0
#         mu1=0
#         for i in range(T,L):
#             P1+=h[i]
#             mu1+=i*h[i]
#         mu1=mu1/(P1+eps)
#         criteriu[T]=P0*mu0*mu0+P1*mu1*mu1
#     THR = np.argmax(criteriu)
#     return THR

# ##calcul histograma pt alegerea pragurilor
# h1,w1=np.shape(img_plan)
# hist1=histograma(h1,w1,img_plan)
# plt.figure("Histograma imaginii zgomotoase"),plt.plot(hist1),plt.title("Histograma imaginii zgomotoase"),plt.show()


# prag = prag_Otsu(hist1)
# SEGM=(img_plan>prag)    #in ex nostru cu biscuiti avem mai mult fundal alb deci folosim <= sa facem fundalul negru si biscuitii albi
# SEGM=np.uint8(SEGM)
# plt.figure(),plt.imshow(SEGM,cmap='gray'),plt.title("Imagine biscuiti segmentata cu prag Otsu"),plt.show()


# [LabelImage, nums]=measure.label(SEGM,return_num='True')

# # Calcularea proprietăților regiunilor
# ALLPROPS = measure.regionprops(LabelImage)
# print(len(ALLPROPS))

# # Extragem aria și perimetrul fiecărei regiuni
# area = np.zeros((nums, 1))
# perimeter = np.zeros((nums, 1))


# new_img=np.zeros([h,w]) 

# size=len(ALLPROPS)
# print(size)

# features = np.zeros([nums,1]) 

# for i in range(nums):
#         features[i] = (ALLPROPS[i].perimeter**2)/(ALLPROPS[i].area*4*np.pi)

# # Algoritm de clustering K-means
# K = 2   
# kmeans = cluster.KMeans(n_clusters=K, random_state=0).fit(features)
# etichete = kmeans.labels_
# centroizi = kmeans.cluster_centers_
# print(centroizi)

# # Creăm o imagine colorată pentru a vizualiza etichetele
# output_image = np.zeros_like(LabelImage, dtype=np.uint8)
# for i in range (nums):
#     if(i==0 or i==1 or i==2 or i==3 or i==4 or i==10 or i==11 or i==12):
#         output_image[LabelImage==(i+1)]=etichete[i]+1
        
# # Afișăm rezultatul final
# plt.figure(), plt.imshow(output_image, cmap="jet",interpolation='none'), plt.colorbar(), plt.title("Rezultatul Clustering"), plt.show()



#################################################### EX 4

# binarizare img
BW=np.uint8(img_plan>50)
plt.figure(),plt.imshow(BW,cmap="gray"),plt.colorbar(),plt.show()

[LabelImage, nums]=measure.label(BW,return_num='True')
# Calcularea proprietăților regiunilor
ALLPROPS = measure.regionprops(LabelImage)

lista_arie = []
for i in range(nums):
    dummy = ALLPROPS[i]
    lista_arie.append(dummy.area)
print(lista_arie)#aria tuturor elementelor
print(max(lista_arie))#aria maxima 
eticheta_ariemax = np.argmax(lista_arie)+1 #(indexul etichetei cu aria maxima)
print(eticheta_ariemax)

componenta_extrasa=eticheta_ariemax
COMP=(LabelImage==componenta_extrasa)
plt.figure(),plt.imshow(COMP,cmap="gray"),plt.colorbar(),plt.show()

supr_tot=h1*w1
print(supr_tot)
aria_max=max(lista_arie)

componenta_extrasa=eticheta_ariemax
COMP=(LabelImage==componenta_extrasa)


dims=np.shape(COMP)
print(dims)
Lines=dims[0]
Columns=dims[1]
Y=COMP[0:Lines,0:Columns] #copie la imagine

#bordare pe ultima linie si coloana
YY=np.zeros([Lines+1, Columns+1])
YY[0:Lines,0:Columns]=Y
YY[Lines,0:Columns]=Y[Lines-1,:]
YY[0:Lines,Columns]=Y[:,Columns-1]
YY[Lines,Columns]=Y[Lines-1,Columns-1]

#afisarea imaginii bordate
#plt.figure(),plt.imshow(YY,cmap='gray'),plt.colorbar(),plt.show()

fx=np.zeros([Lines, Columns])    # fx este componenta verticală.
fy=np.zeros([Lines, Columns])    # fy este componenta orizontală.

for l in range(0,Lines):
    for c in range(0,Columns):
        fx[l,c]=YY[l,c]-YY[l+1,c]
        
for l in range(0,Lines):
    for c in range(0,Columns):        
        fy[l,c]=YY[l,c]-YY[l,c+1]
    
gradY=np.abs(fx)+np.abs(fy)
plt.figure(),plt.imshow(gradY,cmap='gray'),plt.colorbar(),plt.show() #evidentiaza tot conturul

