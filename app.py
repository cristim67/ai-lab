import matplotlib.pyplot as plt
import numpy as np
from skimage import color, io

plt.close('all')

img1 = io.imread('boabe-piper.jpg')
print(type(img1))
print(img1.dtype)
print(np.shape(img1))

plt.figure(),plt.imshow(img1),plt.colorbar(),plt.show()

# convert to greyscale
img_gray = color.rgb2gray(img1)
img_gray = (img_gray * 255).astype(np.uint8)

# histogram
hist = np.zeros(256)

def histogram(h, w, img):
    hist = np.zeros(256)
    for i in range(h):
        for j in range(w):
            hist[img[i, j]] += 1
    return hist / (h * w)   

h1, w1 = np.shape(img_gray) 

hist = histogram(h1, w1, img_gray)

plt.figure(), plt.plot(hist), plt.title("Histogram"), plt.show()