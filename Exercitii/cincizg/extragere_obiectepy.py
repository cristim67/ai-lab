import matplotlib.pyplot as plt
import numpy as np
from skimage import measure, feature, io
from sklearn import cluster
from scipy import signal
L=256 #nr niveluri

#citesc imaginea
img = io.imread('cincizg.bmp')

#vad de ce tip este img
print("img shape si type: ",np.shape(img), img.dtype)

#afisez imaginile
plt.figure("Img originala"),plt.imshow(img,cmap="gray"),plt.colorbar(),plt.show()


# creez histograma
def histograma(img):
    #param: img pt care vrem histograma
    h, w = np.shape(img)
    hist = np.zeros(256)
    for i in range(h):
        for j in range(w):
            hist[img[i,j]] +=1
    #normam histograma la dimensiunile imaginii
    hist = hist / (h*w)
    return hist

# afisare histograma
hist_img = histograma(img)
#hist_img = histograma(img[:,:,1]) 
plt.figure("Histograma imaginii"), plt.plot(hist_img), plt.title("Histograma")


# Segmentare
def prag_Ridler(h, Tcalc):
    # parametrii: histograma, prag calculat
    eps=0.00000001   #pt medie -: ca sa nu impart la 0
    T=0
    while T!=Tcalc: 
        T = Tcalc
        P0=0
        mu0=0
        for i in range(0,T): #pt partea din stanga
            P0+=h[i]         #P0 egalez cu prob de aparitie a nivelului de gri
            mu0+=i*h[i]      #mu0 egalez nivelul de gri * prob de aparitie a lui
        mu0=mu0/(P0+eps)     #: ca sa nu impart la 0
        P1=0
        mu1=0
        for i in range(T,L): #pt partea din dreapta
            P1+=h[i]         #P1 egalez cu prob de aparitie a nivelului de gri
            mu1+=i*h[i]      #mu1 egalez nivelul de gri * prob de aparitie a lui
        mu1=mu1/(P1+eps)      #: ca sa nu impart la 0
        Tcalc = (mu0+mu1)/2
        Tcalc = int(Tcalc)
    return Tcalc 

H, W=np.shape(img)
Y = img[0:H,0:W]
h,_=np.histogram(Y,bins=256,range=(0,256),density=True)

# aplic prag_Ridler pe imagine
Tcalc = 128
THR = prag_Ridler(h,Tcalc) #ca sa obtin pragul folosesc functia prag_Ridler
print("THR: ",THR)
plt.figure("Img >= prag"),plt.imshow(np.uint8(Y>=THR),cmap='gray'),plt.show() 
#pt binar : imaginea segmentata inseamna Y<=THR


# Etichetare
def etichetare(binary_img, kernel_size=None, show=True):
    img_new = binary_img.copy()
    if kernel_size:
        ignore = kernel_size // 2
        for i in range(ignore, binary_img.shape[0] - ignore):
            for j in range(ignore, binary_img.shape[1] - ignore):
                img_new[i, j] = np.median(binary_img[i-ignore:i+ignore+1, j-ignore:j+ignore+1], axis=None)
    
    imagine_etichetata = measure.label(img_new)
    if show:
        plt.figure("Img etichetata"), plt.imshow(imagine_etichetata), plt.colorbar(), plt.title("Imagine etichetata"), plt.show()
    print("Nr obiecte:", np.max(imagine_etichetata))
    return imagine_etichetata


img_label = etichetare((Y>THR), 2)  #img dupa ridler


def extragere_info_obiect(img_label):
    trasaturi_elemente = measure.regionprops(img_label)

    data = [] 
    for i, element in enumerate(trasaturi_elemente):
        bbox_cord = element.bbox
        high = bbox_cord[2] - bbox_cord[0]
        compact = element.perimeter**2 / (4*np.pi*element.area)
        print(i+1, high, compact)
        data.append([high, compact])
    data = np.array(data)
    return data    
data = extragere_info_obiect(img_label)
