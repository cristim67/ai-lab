import math

import matplotlib.pyplot as plt
import numpy as np

# from fcmeans import FCM
import skfuzzy as fuzz
from skimage import color, io, measure

##aducerea img si plotarea ei si a hist\
L=256  
img = io.imread('cincizg.bmp')
# img = np.uint8(255 * color.rgb2gray(img))

print(img.shape)
plt.figure(), plt.imshow(img, cmap='gray', vmin=0, vmax=255), plt.colorbar(), plt.show()

def histograma_imagine(img, plot=False):
    hist, bins = np.histogram(img, bins=256, range=(0, 256), density=False)
    hist = hist / (img.shape[0] * img.shape[1])
    
    if plot:
        plt.figure(), plt.plot(bins[:-1], hist), plt.show()
    
    return hist

hist = histograma_imagine(img, True)


###segmentarea pe prag automat\
# dims=np.shape(img)
# H=dims[0]
# W=dims[1]    
# if len(img.shape)==3:
#     img= np.uint8(0.3*img[:,:,0]+0.6*img[:,:,1] +0.1*img[:,:,2])
# Y=img[0:H,0:W]
# ##alegerea pragului
# t=10
# tcrt=0
# h=np.zeros(L)
# for l in range (0,H):
#   for c in range (0,W):
#     h[Y[l,c]]+=1

# h=h/np.sum(h)

# while t!=tcrt:
#     t=tcrt
#     nr1=nr12=nr2=nr22=0.0001
#     m0=m1=0.00001
#     for i in range(0,L-1):
#        if i<=t-1:
#            nr1= nr1+i*h[i]
#            nr12= nr12+h[i]
#        else:
#            nr2= i*h[i]+nr2
#            nr22= nr22+h[i]
#     m0=nr1/nr12
#     m1=nr2/nr22
#     tcrt=math.floor((m0+m1)/2)

# print(tcrt)
# img_seg= (Y<tcrt)
# plt.figure(),plt.imshow(img_seg, vmin=0,vmax=1), plt.colorbar(), plt.show()
def segmentare_1_prag(img, T):
    img_seg = np.zeros(img.shape)
    img_seg[img > T] = 1
    
    return np.uint8(img_seg)

def segmentare_Riddler(img):
    T = 128

    h = histograma_imagine(img)
    ok = True
    while ok:
        u0, u1 = 0, 0
        for i in range(len(h)):
            if i < T:
                u0 += i * h[i]
            else:
                u1 += i * h[i]
            
    
        u0 /= np.sum(h[:T])
        u1 /= np.sum(h[T:])
    
        T_crt = int((u0 + u1) / 2)
        
        if np.abs(T - T_crt) <= 1:
            ok = False
        else:
            T = T_crt
            
    img_seg = segmentare_1_prag(img, T)
    return img_seg, T    


img_riddler, T_riddler = segmentare_Riddler(img)
print('Riddler:', T_riddler)
plt.figure(), plt.imshow(img_riddler, cmap='gray', vmin=0, vmax=1), plt.colorbar(), plt.show()

#### sa se extraga cel mai mare obiect din img in alta img
# def etichetare(img_seg, plot=False): 
#     LabelImage, nums = measure.label(img_seg, return_num='True')
    
#     if plot:
#         plt.figure(), plt.imshow(LabelImage, cmap="jet", interpolation='none'), plt.colorbar(), plt.show()
        
#     return LabelImage, nums

# img_label, nums = etichetare(img_seg, True)

# allprops = measure.regionprops(img_label)
# biggest_obj = np.argmax([obj.area for obj in allprops])
# print(biggest_obj)



def etichetare(binary_img, kernel_size=None, show=True):
    """
    Primeste o imagine binara, eventual dimensiunea unui filtru median.
    Returneaza imaginea etichetata (+afisare) (+print nr obiecte gasite)
    """
    
    new_img = binary_img.copy()
    if kernel_size:
        ignore = kernel_size // 2
        for i in range(ignore, binary_img.shape[0] - ignore):
            for j in range(ignore, binary_img.shape[1] - ignore):
                new_img[i, j] = np.median(binary_img[i-ignore:i+ignore+1, j-ignore:j+ignore+1], axis=None)
    
    
    imagine_etichetata = measure.label(new_img)
    if show:
        plt.figure(), plt.imshow(imagine_etichetata), plt.colorbar(), plt.title("img etichetata"), plt.show()
    print("Numar obiecte:", np.max(imagine_etichetata))
    return imagine_etichetata

img_label = etichetare(img_riddler, 2)



def extract_infos_object(img_label):
    trasaturi_elemente = measure.regionprops(img_label)

    data = [] ### ATENTIE CE TRASATURI EXTRAGI AICI CA TREBUIE MODIFICAT
    for i, element in enumerate(trasaturi_elemente):
        bbox_cord = element.bbox
        high = bbox_cord[2] - bbox_cord[0]
        compact = element.perimeter**2 / (4*np.pi*element.area)
        print(i+1, high, compact)
        data.append([high, compact])
    data = np.array(data)
    return data
    
data = extract_infos_object(img_label)