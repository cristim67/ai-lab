from sklearn import cluster
import numpy as np
from skimage import io, color, feature, measure
import matplotlib.pyplot as plt


plt.close("all")
L = 256 #numarul initial de niveluri de gri
d = 16 #de cate ori micsorez numarul de niveluri de gri

def texture_features_GCM(img):
    result = feature.graycomatrix(img, [1], [0,  np.pi/4,  np.pi/2], levels=int(L/d))
    fc = feature.graycoprops(result, prop = 'contrast')
    fd = feature.graycoprops(result, prop = 'dissimilarity')
    fh = feature.graycoprops(result, prop = 'homogeneity') 
    fa = feature.graycoprops(result, prop = 'ASM') 
    fe = feature.graycoprops(result, prop = 'energy') 
    fcc = feature.graycoprops(result, prop = 'correlation') 
    
    feat = np.concatenate((fc,fd,fh,fa,fe,fcc), axis = 1)
    return feat



image = io.imread('test1.png')
dims = np.shape(image)
plt.imshow(image, cmap = 'gray'), plt.colorbar(), plt.show()

# se micsoreaza numarul de niveluri de gri
image = np.uint8(image/d)
plt.figure(),plt.imshow(image, cmap = 'gray'), plt.colorbar(), plt.show()

print(np.shape(image))

#image[:,512:1023]=10
#plt.figure(),plt.imshow(image, cmap = 'gray'), plt.colorbar(), plt.show()

# impart imaginea in blocuri (suprapuse 50%) de size_patch x size_patch
size_patch = 16
step = int(size_patch/2)
nr_patch = int(dims[0]/step - 1)  

features1 = np.zeros([nr_patch*nr_patch,18])

index = 0
for i in range(nr_patch):
    for j in range(nr_patch):
        
        img = image[i*step:(i+2)*step, j*step:(j+2)*step]
#        features1[i*nr_patch+j, :] = texture_features_GCM(img)
        features1[index, :] = texture_features_GCM(img)
        index +=  1

#         
print(np.shape(features1))


kmeans = cluster.KMeans(n_clusters=2, random_state=0).fit(features1)
etichete = kmeans.labels_
centroizi = kmeans.cluster_centers_

plt.figure(),plt.imshow(np.reshape(etichete,(nr_patch, nr_patch)), cmap = 'gray'),plt.show()
bw=np.reshape(etichete,(nr_patch, nr_patch))
           
              
[LabelImage, nums]=measure.label(bw,return_num='True')


print(nums)
plt.imshow(LabelImage,cmap="jet"),plt.colorbar(),plt.show()

ALLPROPS=measure.regionprops(LabelImage)
area=np.zeros((nums,1))
for i in range(nums):
    area[i]=ALLPROPS[i].area

new_img=np.zeros(np.shape(LabelImage))
for i in range(nums):
    if (area[i]==np.max(area)):
        new_img[LabelImage==i+1]=1
plt.figure(),plt.imshow(new_img,cmap="jet",interpolation='none'),plt.colorbar(),plt.show()

    
