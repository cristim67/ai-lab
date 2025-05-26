#1) Formati o img de dim 210x210 cu 3 bare orizontale de dim egale a caror intensitati de gri sunt in ordine de sus in jos: 50,100,150. 
#Adaugati zgomot de dispersie 15 si alegeti pragurile de segmentare (manuala) corespunzatoare a.i. bara de sus sa aiba eticheta 1,
#bara din mijloc sa aiba eticheta 2 si bara de jos eticheta 0. Afisati imaginea rezultata.
#2) Pt img segmentata la 1) afisati procentul de pixeli asociat barii situate cel mai jos detectati gresit.
#3) Pt img S3.png folositi convenabil perimetrul si aria a.i. sa delimitati stelutele de restul formelor cu ajutorul unui algoritm de clustering.
#Afisati rezultatul final
#4) Afisati componenta verticala a conturului asociat obiectului care ocupa cel mai mult din imagine daca aceasta repr cel putin 10% din suprafata totala.
#In caz contrar afisati componenta orizontala.

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from skimage import color, io, measure
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
disp=15
img1 = np.zeros([210, 210], dtype='uint8')
img1[:70, :] = 50
img1[70:140, :] = 100
img1[140:, :] = 150

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
T1=75
T2=125
img1_segm=np.uint8(np.zeros([h1,w1]))
for i in range(0,h1):
    for j in range(0,w1):
        
        if (img1_zg[i,j]<T1):
            img1_segm[i,j]=1                  #bara de sus eticheta 1
        elif (T1<=img1_zg[i,j]<=T2):
            img1_segm[i,j]=2                  
        elif (img1_zg[i,j]>T2):
            img1_segm[i,j]=0

print(img1_segm.dtype)
plt.figure(),plt.imshow(img1_segm,cmap="gray"), plt.title(" Img1 segmentata"),plt.colorbar(),plt.show()




########################################### EX2
pixeli_bara_jos = 70 * 210        #fiecare bara are 70 de pixeli latime si 210 lungime
#bara de jos are eticheta 0 si de aceea se scrie !=0
pixeli_gresiti = np.sum(img1_segm[140:, :] != 0)   #bara de jos e de la 140-210 pixeli
print(pixeli_gresiti / pixeli_bara_jos * 100, '%')



############################################ EX3
img2= io.imread('s3poza.png')
print(np.shape(img2))
plt.figure(),plt.imshow(img2,cmap="gray"),plt.title("S3 originala"),plt.colorbar(),plt.show()

img2_plan=img2[:,:,1]
h2,w2=np.shape(img2_plan)

# binarizare img
BW=np.uint8(img2_plan>50)
plt.figure(),plt.imshow(BW,cmap="gray"),plt.colorbar(),plt.show()

[LabelImage, nums]=measure.label(BW,return_num='True')
print(nums)
#plt.figure(figsize=(10,20)),plt.imshow(LabelImage,cmap="jet"),plt.colorbar(),plt.show()
#plt.figure(figsize=(10,20)),plt.imshow(LabelImage,cmap="jet",interpolation='none'),plt.colorbar(),plt.show()


# Calcularea proprietăților regiunilor
ALLPROPS = measure.regionprops(LabelImage)

# Extragem aria și perimetrul fiecărei regiuni
area = np.zeros((nums, 1))
perimeter = np.zeros((nums, 1))
features=np.zeros((nums,1))


for i in range(nums):
    area[i] = ALLPROPS[i].area
    perimeter[i] = ALLPROPS[i].perimeter
    features[i] = area[i]/perimeter[i]


# Concatenam aria și perimetrul într-o singură matrice de caracteristici
#features = np.concatenate((area, perimeter))

# Algoritm de clustering K-means
K = 2  # avem 2 tipuri de obiecte cautate, stelute si non-stelute
kmeans = cluster.KMeans(n_clusters=K, random_state=0).fit(features)
etichete = kmeans.labels_
centroizi = kmeans.cluster_centers_
print(centroizi)

# Creăm o imagine colorată pentru a vizualiza etichetele
output_image = np.zeros_like(LabelImage, dtype=np.uint8)
for i in range(nums):
    if(etichete[i]==1):
        output_image[LabelImage == (i + 1)] = etichete[i] + 1  # Adăugăm 1 pentru a evita zero-ul din fundal

# Afișăm rezultatul final
plt.figure(),plt.imshow(output_image,cmap="jet"),plt.colorbar(),plt.title("Imgine definire stelute"),plt.show()



############################################ EX4

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

supr_tot=h2*w2
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

fx=np.zeros([Lines, Columns])    # fx este componenta orizontala
fy=np.zeros([Lines, Columns])    # fy este componenta verticala.

for l in range(0,Lines):
    for c in range(0,Columns):
        fx[l,c]=YY[l,c]-YY[l+1,c]
        
for l in range(0,Lines):
    for c in range(0,Columns):        
        fy[l,c]=YY[l,c]-YY[l,c+1]

if (aria_max>=0.1*supr_tot):
    plt.figure(),plt.imshow(COMP,cmap="gray"),plt.colorbar(),plt.show()
    #afisati componenta verticala
    plt.figure(),plt.imshow(fx,cmap='gray'),plt.colorbar(),plt.show()
else:
    print("FALS")
    #afisati componenta orizontala
    plt.figure(),plt.imshow(fy,cmap='gray'),plt.colorbar(),plt.show()
    












