import numpy as np
from skimage import io, color, measure
import matplotlib.pyplot as plt

#%% 1.segmentare cu praguri alese manual pt extragere obiecte deschise

#citesc imaginea
img = io.imread('unu.jpeg')
img = np.uint8(255 * color.rgb2gray(img))

#vad de ce tip este img
print("img shape si type: ",np.shape(img), img.dtype)

#afisez imaginea
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
# hist_img = histograma(img[:,:,1]) 
plt.figure("Histograma imaginii"), plt.plot(hist_img), plt.title("Histograma")



# Segmentare cu un singur prag
T = 150    # T = prag de segmentare
img_binar=(img >= T)
plt.figure("Img binarizata"),plt.imshow(img_binar,cmap="gray"),plt.colorbar(),plt.show()

#%% 2. Sa se extraga automat cel mai lung obiect si a se calculeze si afiseze conturul

def etichetare(img_seg, plot=False):         # FUNDAL NEGRU
    LabelImage, nums = measure.label(img_seg, return_num='True')
    if plot:
        plt.figure(), plt.imshow(LabelImage, cmap="jet", interpolation='none'), plt.colorbar(), plt.show()
    return LabelImage, nums

img_label, nums = etichetare(img_binar, True)


allprops = measure.regionprops(img_label)
longest_obj = np.argmax([obj.axis_major_length for obj in allprops])
print("longest_obj: ", longest_obj)


def extragere_obiecte(LabelImage, lista_obiecte):
    img_obiecte = np.zeros(LabelImage.shape)
    
    for i, obiect in enumerate(lista_obiecte):
        img_obiecte[LabelImage == obiect] = i + 1
    return np.uint8(img_obiecte)


img_obiect = extragere_obiecte(img_label, [longest_obj + 1])
plt.figure("Cel mai lung obiect"), plt.imshow(img_obiect, cmap='jet', interpolation='none'), plt.colorbar(), plt.show()


#%% 3.cate obiecte ocupa mai putin de 1.25% din imagine si sa se extraga acele obiecte in alta img

small_objs = [i + 1 for i,obj in enumerate(allprops) if obj.area < (0.0125 * img.shape[0] * img.shape[1])]
print("obiecte mici: ", len(small_objs))

img_small_objs = extragere_obiecte(img_label, small_objs)
plt.figure("Imn noua cu obiecte mici"), plt.imshow(img_small_objs,cmap='jet',interpolation='none'), plt.colorbar(), plt.show()
