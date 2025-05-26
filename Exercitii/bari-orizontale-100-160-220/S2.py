#1) Formati o img de dim 180x180 cu 3 bare verticale de dim egale a caror intensitati de gri sunt in ordine de la stanga la dreapta: 100,160,220. 
#Adaugati zgomot de dispersie 12 si alegeti pragurile de segmentare (manuala) corespunzatoare a.i. bara din stanga sa aiba eticheta 1,
#bara din mijloc sa aiba eticheta 2 si bara din dreapta eticheta 0. Afisati imaginea segmentata.
#2) Pt img segmentata la 1) afisati procentul de pixeli asociat barii din stanga detectati corect. 
#(cati pixeli care apartin barii din stanga sunt detectati corect)
#3) Pt img S2.png folositi o metoda automata prin care sa ramaneti doar cu dreptunghiuri si cercuri.
#Afisati intr-o figura noua rezultatul.
#Etichetati din nou imaginea rezultata a.i. sa nu existe etichete lipsa. 
#Pt aceasta folositi raportul de compactitate R=P**2/(4*pi*A) a.i. sa separati dreptunghiurile de cercuri printr-un procedeu de clustering.
#Afisati rezultatul final.
#4) Det automat si afisati conturul obiectului cel mai aproape de un "cerc ideal".

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
disp=12
img1 = np.zeros([180, 180], dtype='uint8')
img1[:, :60] = 100
img1[:, 60:120] = 160
img1[:, 120:] = 220

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
T1=130
T2=190
img1_segm=np.uint8(np.zeros([h1,w1]))
for i in range(0,h1):
    for j in range(0,w1):
        
        if (img1_zg[i,j]<T1):
            img1_segm[i,j]=1                  #bara din stanga eticheta 1
        elif (T1<=img1_zg[i,j]<=T2):
            img1_segm[i,j]=2                  
        elif (img1_zg[i,j]>T2):
            img1_segm[i,j]=0

print(img1_segm.dtype)
plt.figure(),plt.imshow(img1_segm,cmap="gray"), plt.title(" Img1 segmentata"),plt.colorbar(),plt.show()



########################################### EX2
pixeli_bara_stanga = 60 * 180        #fiecare bara are 60 de pixeli latime si 1800 lungime
#bara din stanga are eticheta 1 si de aceea se scrie ==0
pixeli_corecti = np.sum(img1_segm[:, :60] == 1)   #bara din stanga e de la 0-60 pixeli
print(pixeli_corecti / pixeli_bara_stanga * 100, '%')


########################################### EX3
img2= io.imread('s2.png')
print(np.shape(img2))
plt.figure(),plt.imshow(img2,cmap="gray"),plt.title("s2 originala"),plt.colorbar(),plt.show()

# de verificat sa nu cumva sa fie imaginea color => o fac uint8 gri
# daca imaginea este gri => o facem uint8

img2_grey = np.uint8(255*color.rgb2gray(img2))

plt.figure(), plt.imshow(img2_grey,cmap='gray',vmin=0,vmax=255), plt.title("s2 grey"),plt.colorbar(),plt.show()


#calcul histograma 
h2,w2=np.shape(img2_grey)
hist2=histograma(h2,w2,img2_grey)
plt.figure("Histograma img2_grey"),plt.plot(hist2),plt.title("Histograma img2_grey"),plt.show()

# binarizare img
BW=np.uint8(img2_grey>75)
plt.figure(),plt.imshow(BW,cmap="gray"),plt.colorbar(),plt.show()

#etichetare
[LabelImage, nums]=measure.label(BW,return_num='True')
print(nums)
#plt.figure(figsize=(10,20)),plt.imshow(LabelImage,cmap="jet"),plt.colorbar(),plt.show()
#face un antialias pe zonele de contur
#plt.figure(figsize=(10,20)),plt.imshow(LabelImage,cmap="jet",interpolation='none'),plt.colorbar(),plt.show()


# Calcularea proprietăților regiunilor
ALLPROPS = measure.regionprops(LabelImage)

# Extragem aria și perimetrul fiecărei regiuni
features = np.zeros([nums, 1])

for i in range(nums):
    features[i] = (ALLPROPS[i].perimeter**2)/(ALLPROPS[i].area*4*np.pi)

# Concatenam aria și perimetrul într-o singură matrice de caracteristici
#features = np.concatenate((features))


# Algoritm de clustering K-means
K = 2  # avem 2 tipuri de obiecte cautate, cercuri si dreptunghiuri
kmeans = cluster.KMeans(n_clusters=K, random_state=0).fit(features)
etichete = kmeans.labels_
centroizi = kmeans.cluster_centers_
print(centroizi)

# Creăm o imagine colorată pentru a vizualiza etichetele
output_image = np.zeros_like(LabelImage, dtype=np.uint8)
for i in range(nums):
    output_image[LabelImage == (i + 1)] = etichete[i] + 1  # Adăugăm 1 pentru a evita zero-ul din fundal

# Afișăm rezultatul final
plt.figure(),plt.imshow(output_image,cmap="jet", interpolation='none'),plt.colorbar(),plt.title("Rezultat clustering"),plt.show()



########################################### EX4
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








