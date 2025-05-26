import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from skimage import color, feature, io, measure
from sklearn import cluster

plt.close('all')


######################### EX1
img= io.imread('s6.jpg')
print(np.shape(img))
plt.figure(),plt.imshow(img,cmap="gray"),plt.title("Img originala"),plt.colorbar(),plt.show()

def histograma(h,w,img):
    hist=np.zeros(256)
    for i in range(h):
        for j in range(w):
            hist[img[i,j]]+=1    
    return hist/(h*w)

#calcul histograma pt alegerea pragurilor
h1,w1=np.shape(img)
hist=histograma(h1,w1,img)
plt.figure(),plt.plot(hist),plt.title("Histograma img"),plt.show()#calcul histograma pt alegerea pragurilor

#segmentare manuala
#alegem pragurile (ne uitam pe histograma pt a le alege)
T1=70
T2=180
img1_segm=np.uint8(np.zeros([h1,w1]))
for i in range(0,h1):
    for j in range(0,w1):
        
        if (img[i,j]<T1):
            img1_segm[i,j]=2                  
        elif (T1<=img[i,j]<=T2):
            img1_segm[i,j]=1                  
        elif (img[i,j]>T2):
            img1_segm[i,j]=0

print(img1_segm.dtype)
plt.figure(),plt.imshow(img1_segm,cmap="gray"), plt.title(" Img segmentata"),plt.colorbar(),plt.show()




############################################ EX2
L=256
img2 = np.zeros([150, 150], dtype='uint8')
img2[:75, :] = 100
img2[75:, :] = 200

##afisare img originala  
plt.figure(), plt.imshow(img2, cmap='gray', vmin=0, vmax=255), plt.colorbar(), plt.title("Imagine initiala"), plt.show()


def add_noise(Y,disp):
    Yzg=Y+np.random.normal(0,disp,[Lines,Columns])
    for i in range(0,Lines):
        for j in range(0, Columns):
            if (Yzg[i,j]>L-1):
                Yzg[i,j]=L-1
            else:
                if (Yzg[i,j]<0):
                    Yzg[i,j]=0

    return np.uint8(Yzg)


MASK = np.zeros([150,150])
MASK[99:101,:]=1

#metoda gradient
dims=np.shape(img2)
Lines=dims[0]
Columns=dims[1]
Y=img2[0:Lines,0:Columns] #copie la imagine


noise_image = add_noise(Y,50)
Yy=noise_image[0:Lines,0:Columns] #copie la imagine

#bordare pe ultima linie si coloana
YY=np.zeros([Lines+1, Columns+1])
YY[0:Lines,0:Columns]=Yy
YY[Lines,0:Columns]=Yy[Lines-1,:]
YY[0:Lines,Columns]=Yy[:,Columns-1]
YY[Lines,Columns]=Yy[Lines-1,Columns-1]

#afisarea imaginii bordate
plt.figure(),plt.imshow(YY,cmap='gray'),plt.colorbar(),plt.title("Imagine bordata"),plt.show()

fx=np.zeros([Lines, Columns])
fy=np.zeros([Lines, Columns])

for l in range(0,Lines):
    for c in range(0,Columns):
        fx[l,c]=YY[l,c]-YY[l+1,c]
        
for l in range(0,Lines):
    for c in range(0,Columns):        
        fy[l,c]=YY[l,c]-YY[l,c+1]


#diferenta intre nivelurile de gri pe cele 2 directi        
plt.figure(),plt.imshow(fx,cmap='gray'),plt.colorbar(), plt.title("Diferenta dintre nivelurile de gri pe fx"),plt.show()
plt.figure(),plt.imshow(fy,cmap='gray'),plt.colorbar(),plt.title("Diferenta dintre nivelurile de gri pe fy"),plt.show()

#afisarea gradientului imaginii
gradY=np.abs(fx)+np.abs(fy)
print(gradY)
plt.figure(),plt.imshow(gradY,cmap='gray'),plt.colorbar(),plt.title("Afisarea gradientului imaginii"), plt.show()

# Harta binara de contururi(prin binarizare)
plt.figure(),plt.imshow(gradY>50,cmap='gray'),plt.colorbar(),plt.show()


def my_precrec(MASK, SEGM):
    epsilon=0.00001
    TP=SEGM*MASK
    FP=SEGM*(1-MASK)
    FN=(1-SEGM)*MASK
    prec=np.sum(TP)/(np.sum(TP)+np.sum(FP)+epsilon)
    rec=np.sum(TP)/(np.sum(TP)+np.sum(FN)+epsilon)
    return prec, rec

#exemplu de segmentare si urmat de maximizarea preciziei si reamintirii

fx=signal.convolve2d(noise_image,np.array([[0,-1,0],[0,0,0],[0,1,0]]),boundary='symm', mode='same')
fy=signal.convolve2d(noise_image,np.array([[0,0,0],[-1,0,1],[0,0,0]]),boundary='symm', mode='same')
gradY=np.abs(fx)+np.abs(fy)
all_prec = np.zeros(np.max(gradY))
all_rec = np.zeros(np.max(gradY))

for i in range(0,np.max(gradY)):
    all_prec[i],all_rec[i] = my_precrec(MASK,gradY>i)
    
diff = np.abs(all_prec-all_rec)
intersectie = np.argmin(diff)
print("Precizia si recall-ul se intersecteaza la "+str(intersectie))
plt.figure(),plt.plot(all_prec),plt.title("precizie"),plt.show()
plt.figure(),plt.plot(all_rec),plt.title("recall"),plt.show()
plt.figure(),plt.plot(all_rec),plt.plot(all_prec),plt.show()#cu albastru e recall, cu galben e precizie
plt.figure(),plt.imshow(gradY>intersectie,cmap='gray'),plt.colorbar(),plt.show()



################################### EX3
L = 256 #numarul initial de niveluri de gri
d = 16 #de cate ori micsorez numarul de niveluri de gri
def texture_features_GCM(img):
    result = feature.graycomatrix(img, [1], [0,  np.pi/4,  np.pi/2], levels=int(L/d))
    fc = feature.graycoprops(result, prop = 'contrast')
    fh = feature.graycoprops(result, prop = 'homogeneity') 
    fe = feature.graycoprops(result, prop = 'energy') 
    feat = np.concatenate((fc,fh,fe), axis = 1)
    return feat

image = io.imread('test1.png')
dims = np.shape(image)
plt.figure(),plt.imshow(image, cmap = 'gray'), plt.colorbar(), plt.show()

# se micsoreaza numarul de niveluri de gri
image = np.uint8(image/d)
plt.figure(),plt.imshow(image, cmap = 'gray'), plt.colorbar(), plt.show()

# impart imaginea in blocuri (suprapuse 0%) de size_patch x size_patch
size_patch = 16
step = int(size_patch/2) #pentru suprapunere 50%
nr_patch = int(dims[0]/step - 1)  

features1 = np.zeros([nr_patch*nr_patch,9])

index = 0
for i in range(nr_patch):
    for j in range(nr_patch):     
        img = image[i*step:(i+2)*step, j*step:(j+2)*step]
        features1[index, :] = texture_features_GCM(img)
        index +=  1
print(np.shape(features1))

kmeans = cluster.KMeans(n_clusters=2, random_state=0).fit(features1)
etichete = kmeans.labels_
centroizi = kmeans.cluster_centers_

img3 = np.reshape(etichete,(nr_patch, nr_patch))

#plt.figure(),plt.imshow(np.reshape(etichete,(nr_patch, nr_patch)), cmap = 'gray'),plt.colorbar(),plt.show()
plt.figure(),plt.imshow(img3, cmap = 'gray'),plt.colorbar(),plt.show()


############################### EX4
#se considera clusterizarea ca o imagine segmentata
IMG_SEGM=np.reshape(etichete,(nr_patch, nr_patch))
CLASA = np.uint8(IMG_SEGM==0)# pentru a se schimba fundalul in negru
plt.figure(),plt.imshow(CLASA,cmap='gray'),plt.show()

#se face etichetarea propriu-zisa
[LabelImage, nums] = measure.label(CLASA,return_num='True')
print(nums)
plt.figure(),plt.imshow(LabelImage,cmap='jet'),plt.colorbar(),plt.show()


# Calcularea proprietăților regiunilor
ALLPROPS = measure.regionprops(LabelImage)

# Extragem aria și perimetrul fiecărei regiuni
area = np.zeros((nums, 1))

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





