import numpy as np
from skimage import io, color, measure
import matplotlib.pyplot as plt
from scipy import signal
from sklearn import cluster


def add_zgomot_gaussian(img, disp):
    zg = np.random.normal(0, disp, img.shape)
    
    img_zg = np.zeros(img.shape)
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j] + zg[i, j] < 0:
                img_zg[i, j] = 0
            elif img[i, j] + zg[i, j] > 255:
                img_zg[i, j] = 255
            else:
                img_zg[i, j] = img[i, j] + zg[i, j]
                
    return np.uint8(img_zg)


def segmentare_2_prag(img, T1, T2):
    img_seg = np.zeros(img.shape)
    
    mask_under = img < T1
    mask_over = img > T2
    mask_between = (1-mask_under) * (1-mask_over)
    
    img_seg[mask_under == 1]= 1
    img_seg[mask_between == 1] = 2
    img_seg[mask_over == 1] = 0
    
    return np.uint8(img_seg)


def histograma_imagine(img, plot=False):
    hist, bins = np.histogram(img, bins=256, range=(0, 256), density=False)
    hist = hist / (img.shape[0] * img.shape[1])
    
    if plot:
        plt.figure(), plt.plot(bins[:-1], hist), plt.show()
    
    return hist

#### A
###facerti o img 210 pe 210 cu 3 bare oriz cu intesit 50 100 150
##add zg de disp 15 ai alegeti prag segm sus et 1 mij et 2 jos et 0
img1 = np.zeros((210, 210))
img1[:70, :] = 50
img1[70:140, :] = 100
img1[140:, :] = 150
##afisare img originala 
plt.figure(), plt.imshow(img1, cmap='gray', vmin=0, vmax=255), plt.colorbar(), plt.show()
##adaugarea de zgomot sia fisare img zhomot
img1_zg = add_zgomot_gaussian(img1, 15)
plt.figure(), plt.imshow(img1_zg, cmap='gray', vmin=0, vmax=255), plt.colorbar(), plt.show()
hist1_zg = histograma_imagine(img1_zg, True)


img1_seg = segmentare_2_prag(img1_zg, 75, 125)
plt.figure(), plt.imshow(img1_seg, cmap='gray'), plt.colorbar(), plt.show()


#### B
##afis procentul de pix  asociati barii de jos detectati gresit
pixeli_bara_jos = 70 * 210
pixeli_gresiti = np.sum(img1_seg[140:, :] != 0)
print(pixeli_gresiti / pixeli_bara_jos * 100, '%')


### C
##fol persim si arie  ai sa det stelutele si restul formelor
##cu ajutorul alg de clustering  afis rez final
img = io.imread('S3poza.png')
img = np.uint8(255 * color.rgb2gray(img))
plt.figure(), plt.imshow(img, cmap='gray', vmin=0, vmax=255), plt.colorbar(), plt.show()


def etichetare(img_seg, plot=False): # FUNDAL NEGRU
    LabelImage, nums = measure.label(img_seg, return_num='True')
    
    if plot:
        plt.figure(), plt.imshow(LabelImage, cmap="jet", interpolation='none'), plt.colorbar(), plt.show()
        
    return LabelImage, nums

hist = histograma_imagine(img, True)
img[img < 50] = 0
plt.figure(), plt.imshow(img, cmap='gray', vmin=0, vmax=255), plt.colorbar(), plt.show()

img_etichetata, _ = etichetare(img, True)


def kmeans(data, shape, K, plot=False): # DATA = array(exemple, features)
    kmeans = cluster.KMeans(n_clusters=K, random_state=0).fit(data)

    etichete = kmeans.labels_
    centroizi = kmeans.cluster_centers_
    
    LabelImage = etichete.reshape(shape)
    
    if plot:
        plt.figure(), plt.imshow(LabelImage, cmap="jet", interpolation='none'), plt.colorbar(), plt.show()
        
    return LabelImage, centroizi


def clustering_obiecte(img_etichetata, K, plot=False):
    allprops = measure.regionprops(img_etichetata)
    
    # SAU ORICATE ALTE FEATURES SUB ORICE FORMA
    feature_1 = np.array([obj.area for obj in allprops])
    feature_2 = np.array([obj.perimeter for obj in allprops])
    feature_3 = feature_2**2 / feature_1
    feature_4 = feature_2 / feature_1
    
    features = np.vstack([feature_4])
    features = features.transpose()
    
    labels, _ = kmeans(features, len(feature_1), K)
        
    img_cluster = np.zeros(img_etichetata.shape)

    for i in range(img_etichetata.shape[0]):
        for j in range(img_etichetata.shape[1]):
            if img_etichetata[i, j] != 0:
                img_cluster[i, j] = labels[img_etichetata[i, j] - 1] + 1
                
    if plot:
        plt.figure(), plt.imshow(img_cluster, cmap="jet", interpolation='none'), plt.colorbar(), plt.show()

    return img_cluster

img_cluster = clustering_obiecte(img_etichetata, 2, True)


##### D
####afis comp veric care ocupa cea mai mare arie care ocupa cel putin 10% din suprafat tot
allprops = measure.regionprops(img_etichetata)
biggest_obj = np.argmax([obj.area for obj in allprops])
print(biggest_obj)


def extragere_obiecte(LabelImage, lista_obiecte):
    img_obiecte = np.zeros(LabelImage.shape)
    for i, obiect in enumerate(lista_obiecte):
        img_obiecte[LabelImage == obiect] = i + 1
        
    return np.uint8(img_obiecte)

img_obiect = extragere_obiecte(img_etichetata, [biggest_obj + 1])
plt.figure(), plt.imshow(img_obiect, cmap='gray', vmin=0, vmax=1), plt.colorbar(), plt.show()

arie_obiect = np.sum(img_obiect)

if arie_obiect > 0.1 * img.shape[0] * img.shape[1]:
    fx = signal.convolve2d(img_obiect, np.array([[0,1,0],[0,0,0],[0,-1,0]]), boundary='symm', mode='same')
    plt.figure(), plt.imshow(fx, cmap='gray'), plt.colorbar(), plt.show()
else:
    fy = signal.convolve2d(img_obiect, np.array([[1,0,-1],[1,0,-1],[1,0,-1]]), boundary='symm', mode='same')
    plt.figure(), plt.imshow(fy, cmap='gray'), plt.colorbar(), plt.show()





