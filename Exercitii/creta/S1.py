# Se da img "unu.bmp".
#a) Sa se segmenteze imaginea cu praguri de segmentare alese manual, pt a extrage obiectele deschise la culoare
#b) Sa se extraga(in mod automat) obiectul cel mai alungit din imagine si sa i se calculeze si afiseze conturul
#c) Sa se numere cate obiecte ocupa mai putin de 1.25% din suprafata imaginii si sa se extraga aceste obiecte intr-o alta imagine


import matplotlib.pyplot as plt
import numpy as np
from skimage import color, io, measure
from sklearn import cluster

############################### PUNCTUL A
plt.close('all')


def histograma(h,w,img):
    hist=np.zeros(256)
    for i in range(h):
        for j in range(w):
            hist[img[i,j]]+=1    
    return hist/(h*w)


img1= io.imread('unu.jpeg')
print(np.shape(img1))
plt.figure(),plt.imshow(img1,cmap="gray"),plt.title("img1 originala"),plt.colorbar(),plt.show()

img1_grey = np.uint8(255*color.rgb2gray(img1))

plt.figure(), plt.imshow(img1_grey,cmap='gray',vmin=0,vmax=255), plt.title("img1 grey"),plt.colorbar(),plt.show()


#calcul histograma pt alegerea pragurilor
h1,w1=np.shape(img1_grey)
hist=histograma(h1,w1,img1_grey)
plt.figure("Histograma img1"),plt.plot(hist),plt.title("Histograma img"),plt.show()


#segmentare manuala
#alegem pragurile (ne uitam pe histograma pt a le alege)
T=150
img1_segm=np.uint8(np.zeros([h1,w1]))
for i in range(0,h1):
    for j in range(0,w1):
        
        if (img1_grey[i,j]<T):
            img1_segm[i,j]=0                 
        else:
            img1_segm[i,j]=1                 


print(img1_segm.dtype)
plt.figure(),plt.imshow(img1_segm,cmap="gray"), plt.title(" Img1 segmentata"),plt.colorbar(),plt.show()


############################### PUNCTUL B
(h,w)=np.shape(img1_grey)
# binarizare img
BW=np.uint8(img1_grey>150)
#plt.figure(),plt.imshow(BW,cmap="gray"),plt.colorbar(),plt.show()

#etichetare
[LabelImage, nums]=measure.label(BW,return_num='True')
print(nums)
#plt.figure(figsize=(10,20)),plt.imshow(LabelImage,cmap="jet"),plt.colorbar(),plt.show()
#face un antialias pe zonele de contur
#plt.figure(figsize=(10,20)),plt.imshow(LabelImage,cmap="jet",interpolation='none'),plt.colorbar(),plt.show()


# Calcularea proprietăților regiunilor
ALLPROPS = measure.regionprops(LabelImage)

features = np.zeros([nums, 1])

for i in range(nums):
    features[i] = ALLPROPS[i].perimeter
    
# Algoritm de clustering K-means
K = 11  # avem 11 obiecte (10 crete+ 1 fundal)
kmeans = cluster.KMeans(n_clusters=K, random_state=0).fit(features)
etichete = kmeans.labels_
centroizi = kmeans.cluster_centers_
#print(centroizi)

# Creăm o imagine colorată pentru a vizualiza etichetele
output_image = np.zeros_like(LabelImage, dtype=np.uint8)
for i in range(nums):
    output_image[LabelImage == (i + 1)] = etichete[i] + 1  # Adăugăm 1 pentru a evita zero-ul din fundal

# Afișăm rezultatul final cu toate obiectele etichetate
#plt.figure(),plt.imshow(output_image,cmap="jet", interpolation='none'),plt.colorbar(),plt.title("Rezultat clustering"),plt.show()


#pp ca primul obiect are perimetrul maxim
maxim = ALLPROPS[0].perimeter
obiect = 0  #in variabila obiect stocam eticheta obiectului cautat
for i in range(nums):
    cautare =  ALLPROPS[i].perimeter
    if(cautare > maxim):
        maxim = cautare
        obiect =i

# Creăm o imagine colorată pentru a vizualiza etichetele
output_image = np.zeros_like(LabelImage, dtype=np.uint8)
for i in range(nums):
    if(etichete[i]==obiect):
        output_image[LabelImage == (i + 1)] = etichete[i] + 1  # Adăugăm 1 pentru a evita zero-ul din fundal

# Afișăm rezultatul final, doar obiectul cel mai alungit
plt.figure(),plt.imshow(output_image,cmap="gray"),plt.colorbar(),plt.title("Rezultat clustering"),plt.show()


componenta_extrasa = obiect+1
COMP=(LabelImage==componenta_extrasa)

dims=np.shape(COMP)

print(dims)

Lines=dims[0]
Columns=dims[1]

Y=COMP[0:Lines,0:Columns] #copie la imagine

#bordare pe ultima linie si coloana
YY=np.zeros([Lines+1, Columns+1])
YY[0:Lines,0:Columns]=Y               #pune in YY img initiala
YY[Lines,0:Columns]=Y[Lines-1,:]      #copiaza ultima linie
YY[0:Lines,Columns]=Y[:,Columns-1]    #copiaza ultima coloana
YY[Lines,Columns]=Y[Lines-1,Columns-1]   #adauga linia si coloana copiate anterior
print(np.shape(YY))  #s-a marit cu o linie si o coloana

#plt.figure(),plt.imshow(YY,cmap='gray'),plt.colorbar(), plt.title("Img bordata"),plt.show()

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
print(gradY)
plt.figure(),plt.imshow(gradY,cmap='gray'),plt.colorbar(),plt.show() #evidentiaza tot conturul


############################### PUNCTUL C
# Extragem aria fiecărei regiuni
area = np.zeros((nums, 1))

features2 = np.zeros([nums,1]) 
for i in range(nums):
        features2[i] = ALLPROPS[i].area

# Algoritm de clustering K-means
K = 11    #nr de obiecte din imagine (avem 10 crete + 1 fundal)
kmeans = cluster.KMeans(n_clusters=K, random_state=0).fit(features2)
etichete2 = kmeans.labels_
centroizi2 = kmeans.cluster_centers_
#print(centroizi)

# Creăm o imagine colorată pentru a vizualiza etichetele
output_image2 = np.zeros_like(LabelImage, dtype=np.uint8)
for i in range(nums):
    output_image2[LabelImage == (i + 1)] = etichete2[i] + 1  # Adăugăm 1 pentru a evita zero-ul din fundal

# Afișăm rezultatul final
plt.figure(), plt.imshow(output_image2, cmap="jet",interpolation='none'), plt.colorbar(), plt.title("Rezultatul Clustering"), plt.show()

#numaram cate ob
contor=0
arie_totala=300*600
threshold = 0.0125 * arie_totala
lista=[]

new_img=np.zeros((h,w))

for i in range(nums):
    if(features2[i]< threshold):
        contor=contor+1
        obiect=i
        lista.append(i)

for i in range(nums):
    if (features2[i]< threshold):
        new_img[LabelImage==i+1]=1
plt.figure(),plt.imshow(new_img,cmap="gray",interpolation='none'),plt.colorbar(),plt.show()
print(contor)





