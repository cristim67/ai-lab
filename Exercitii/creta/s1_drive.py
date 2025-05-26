import numpy as np
from skimage import io, color, measure
import matplotlib.pyplot as plt


############## A
img = io.imread('unu.jpeg')
img = np.uint8(255 * color.rgb2gray(img))
print(img.shape)
plt.figure(), plt.imshow(img, cmap='gray', vmin=0, vmax=255), plt.colorbar(), plt.show()

def histograma_imagine(img, plot=False):
    hist, bins = np.histogram(img, bins=256, range=(0, 256), density=False)
    hist = hist / (img.shape[0] * img.shape[1])
    
    if plot:
        plt.figure(), plt.plot(bins[:-1], hist), plt.show()
    
    return hist


hist = histograma_imagine(img, True)

def segmentare_1_prag(img, T):
    img_seg = np.zeros(img.shape)
    img_seg[img > T] = 1
    
    return np.uint8(img_seg)

img_seg = segmentare_1_prag(img, 150)
plt.figure(), plt.imshow(img_seg, cmap='gray', vmin=0, vmax=1), plt.colorbar(), plt.show()


################### B
def etichetare(img_seg, plot=False): # FUNDAL NEGRU
    LabelImage, nums = measure.label(img_seg, return_num='True')
    
    if plot:
        plt.figure(), plt.imshow(LabelImage, cmap="jet", interpolation='none'), plt.colorbar(), plt.show()
        
    return LabelImage, nums

img_label, nums = etichetare(img_seg, True)

allprops = measure.regionprops(img_label)
longest_obj = np.argmax([obj.axis_major_length for obj in allprops])
print(longest_obj)

def extragere_obiecte(LabelImage, lista_obiecte):
    img_obiecte = np.zeros(LabelImage.shape)
    for i, obiect in enumerate(lista_obiecte):
        img_obiecte[LabelImage == obiect] = i + 1
        
    return np.uint8(img_obiecte)

img_obiect = extragere_obiecte(img_label, [longest_obj + 1])
plt.figure(), plt.imshow(img_obiect, cmap='jet', interpolation='none'), plt.colorbar(), plt.show()

########################## C
small_objs = [i + 1 for i, obj in enumerate(allprops) if obj.area < 0.0125 * img.shape[0] * img.shape[1]]
print(len(small_objs))

img_small_objs = extragere_obiecte(img_label, small_objs)
plt.figure(), plt.imshow(img_small_objs, cmap='jet', interpolation='none'), plt.colorbar(), plt.show()