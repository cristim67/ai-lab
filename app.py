import math

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

#from fcmeans import FCM
import skfuzzy as fuzz
from numpy import ndarray as npnd
from scipy import signal
from skimage import color, feature, io, measure
from skimage.transform import resize
from sklearn import cluster, datasets

#1) Procesarea imaginii 'rochie.jpg'

# Încărcarea imaginii și afișarea ei
image = io.imread('rochie.jpg')

# Afișăm imaginea originală în tonuri de gri
plt.figure(),plt.imshow(image,cmap="gray"),plt.title("Rochie"),plt.colorbar(),plt.show()

print(np.shape(image))

# Funcție pentru calculul histogramei
# Această funcție calculează distribuția intensităților pixelilor în imagine
# Returnează un vector de 256 de elemente, fiecare reprezentând frecvența relativă a unei intensități
def histogram(height, width, image):
    hist=np.zeros(256)  # Inițializăm histograma cu zerouri
    for i in range(height):
        for j in range(width):
            hist[image[i,j]]+=1    # Incrementăm contorul pentru intensitatea curentă
    return hist/(height*width)    # Normalizăm histograma împărțind la numărul total de pixeli

# Calculăm și afișăm histograma imaginii
height, width = np.shape(image)
hist = histogram(height, width, image)
plt.figure(),plt.plot(hist),plt.title("Histograma Rochie"),plt.show()

# Alegerea pragurilor de segmentare bazate pe histogramă
# Aceste praguri sunt alese pentru a separa regiunile întunecate, medii și luminoase
Threshold1 = 80   # Prag pentru separarea regiunilor întunecate
Threshold2 = 225  # Prag pentru separarea regiunilor luminoase

# Crearea imaginii segmentate
# 0 = regiuni luminoase
# 1 = regiuni medii
# 2 = regiuni întunecate
result = np.zeros((height, width), dtype=np.uint8)

for i in range(height):
    for j in range(width):
        if (image[i,j]<Threshold1):
            result[i,j]=2                 # Regiuni întunecate
        elif (Threshold1<=image[i,j]<=Threshold2):
            result[i,j]=1                 # Regiuni medii
        elif (image[i,j]>Threshold2):
            result[i,j]=0                 # Regiuni luminoase

plt.figure(), plt.imshow(result, cmap='gray')
plt.title("Segmentare rochie"), plt.colorbar(), plt.show()


#2) Procesarea imaginii 'S6.png'

# Încărcarea și conversia imaginii
image2 = io.imread('S6.png')

# Conversia imaginii la tonuri de gri dacă este necesar
# Verificăm dacă imaginea este color (3 canale) sau RGBA (4 canale)
if image2.ndim == 3:
    if image2.shape[2] == 4:  # Dacă imaginea este în format RGBA
        image2 = color.rgba2rgb(image2)  # Mai întâi convertim RGBA la RGB
    image2 = color.rgb2gray(image2)  # Apoi convertim la tonuri de gri

# Calculul gradientului folosind convoluții
# Folosim doi nuclei Sobel pentru a calcula gradientul în direcțiile x și y pentru a detecta marginile.
gradient_x = signal.convolve2d(image2,np.array([[0,-1,0],[0,0,0],[0,1,0]]),boundary='symm', mode='same')
gradient_y = signal.convolve2d(image2,np.array([[0,0,0],[-1,0,1],[0,0,0]]),boundary='symm', mode='same')
gradient = np.abs(gradient_x)+np.abs(gradient_y)  # Gradientul total este suma modulelor
     
plt.figure(),plt.imshow(gradient,cmap='gray'),plt.colorbar(),plt.show()

# Binarizarea gradientului pentru a obține contururile
# Folosim un prag adaptiv bazat pe media gradientului
threshold = np.mean(gradient) * 1.5  # prag adaptiv
contours = gradient > threshold

# Identificarea obiectelor conectate în imagine
# Folosim algoritmul de etichetare pentru a găsi obiecte separate
labeled_objects = measure.label(contours)
regions = measure.regionprops(labeled_objects)

# Calculul distanței între fiecare pereche de obiecte
# Găsim perechea de obiecte care sunt cele mai apropiate
min_distance = float('inf')
closest_pair = None

for i in range(len(regions)):
    for j in range(i + 1, len(regions)):
        # Calculul distanței euclidiene între centroidele obiectelor
        centroid_i = regions[i].centroid
        centroid_j = regions[j].centroid
        distance = np.sqrt((centroid_i[0] - centroid_j[0])**2 + (centroid_i[1] - centroid_j[1])**2)
        
        if distance < min_distance:
            min_distance = distance
            closest_pair = (i + 1, j + 1)  # +1 deoarece etichetele încep de la 1

# Crearea unei noi imagini care conține doar cele două obiecte cele mai apropiate
result2 = np.zeros_like(labeled_objects)
if closest_pair:
    result2[labeled_objects == closest_pair[0]] = 1
    result2[labeled_objects == closest_pair[1]] = 1

# Afișarea imaginii cu obiectele cele mai apropiate
plt.figure(), plt.imshow(result2, cmap='gray')
plt.title("Cele două obiecte cele mai apropiate"), plt.show()

#3) Analiza formelor obiectelor
#Calculăm raportul p**2/A pentru fiecare obiect și le grupăm în categorii.
#Acest raport este invariant la scalare și poate fi folosit pentru clasificarea formelor.

# Calculul raportului r = p**2 / A pentru fiecare obiect
ratios = []
for region in regions:
    perimeter = region.perimeter
    area = region.area
    if area > 0:  # evităm împărțirea la zero
        ratio = (perimeter ** 2) / area
        ratios.append(ratio)

# Gruparea formelor în 3 categorii folosind K-means
if len(ratios) > 0:
    # Convertim rapoartele într-un array 2D pentru K-means
    X = np.array(ratios).reshape(-1, 1)
    
    # Aplicăm K-means pentru 3 categorii
    kmeans = cluster.KMeans(n_clusters=3, random_state=42)
    labels = kmeans.fit_predict(X)
    
    # Creăm o imagine colorată pentru fiecare categorie
    color_result = np.zeros((labeled_objects.shape[0], labeled_objects.shape[1], 3))
    
    # Culori pentru fiecare categorie (RGB)
    colors = [(1,0,0), (0,1,0), (0,0,1)]  # roșu, verde, albastru
    
    # Colorăm fiecare obiect în funcție de categoria sa
    for i, region in enumerate(regions):
        if region.area > 0:
            ratio = (region.perimeter ** 2) / region.area
            category = kmeans.predict([[ratio]])[0]
            mask = labeled_objects == (i + 1)
            for c in range(3):
                color_result[mask, c] = colors[category][c]
    
    # Afișăm rezultatul
    plt.figure(figsize=(10, 8))
    plt.imshow(color_result)
    plt.title("Gruparea formelor în 3 categorii bazată pe raportul r = p**2/A")
    plt.show()

#4) Identificarea obiectului cel mai apropiat de un hexagon ideal
#Calculăm raportul teoretic pentru un hexagon regulat și găsim obiectul cu raportul cel mai apropiat.


# Calculul raportului pentru un hexagon regulat:
# Pentru un hexagon regulat cu latura 'latura':
# - Perimetrul (p) = 6 * latura
# - Aria (A) = (3 * sqrt(3) * latura^2) / 2
# Raportul r = p^2 / A = (6a)^2 / ((3 * sqrt(3) * a^2) / 2)
# r = 36a^2 / ((3 * sqrt(3) * a^2) / 2)
# r = 72 / (3 * sqrt(3))
# r = 24 / sqrt(3)
# obs latura este considerată 1 pentru raportul ideal

latura = 1
ideal_hexagon_ratio = 72 / (3*latura*latura*math.sqrt(3))
print(f"Raportul teoretic pentru un hexagon regulat: {ideal_hexagon_ratio:.2f}")

# Găsim obiectul cu raportul cel mai apropiat de hexagonul ideal
min_difference = float('inf')
hexagon_object = None
object_index = None

for i, region in enumerate(regions):
    if region.area > 0:
        perimeter = region.perimeter
        area = region.area
        ratio = (perimeter ** 2) / area
        difference = abs(ratio - ideal_hexagon_ratio)
        
        if difference < min_difference:
            min_difference = difference
            hexagon_object = region
            object_index = i

# Creăm o imagine pentru a afișa doar obiectul cel mai apropiat de hexagon
hexagon_image = np.zeros_like(labeled_objects)
if hexagon_object is not None:
    hexagon_image[labeled_objects == (object_index + 1)] = 1
    
    # Afișăm rezultatul
    plt.figure(figsize=(10, 8))
    plt.imshow(hexagon_image, cmap='gray')
    plt.title(f"Obiectul cel mai apropiat de un hexagon ideal (raport = {(hexagon_object.perimeter ** 2) / hexagon_object.area:.2f})")
    plt.show()
    
    # Afișăm și conturul obiectului
    contour = measure.find_contours(hexagon_image, 0.5)[0]
    plt.figure(figsize=(10, 8))
    plt.plot(contour[:, 1], contour[:, 0])
    plt.title("Conturul obiectului cel mai apropiat de un hexagon ideal")
    plt.axis('equal')
    plt.show()
    
    # Afișăm diferența față de hexagonul ideal
    print(f"Diferența față de hexagonul ideal: {min_difference:.2f}")