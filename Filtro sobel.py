import cv2
import numpy as np
import matplotlib.pyplot as plt

# Cargar las imágenes usando OpenCV
img1 = cv2.imread('imagenes/imagen7.png', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('imagenes/imagen11.jpg', cv2.IMREAD_GRAYSCALE) 

# Aplicar el filtro Sobel en la dirección X
sobelXImag1 = cv2.Sobel(img1, cv2.CV_64F, 1, 0, ksize=3) # Especifica el formato de los datos de la imagen procesada. En este caso, CV_64F se refiere a un formato de punto flotante de 64 bits
# Aplicar el filtro Sobel en la dirección Y
sobelYImag1 = cv2.Sobel(img1, cv2.CV_64F, 0, 1, ksize=3) # Especifica el formato de los datos de la imagen procesada. En este caso, CV_64F se refiere a un formato de punto flotante de 64 bits

#Procesamiento imagen2
sobelXImag2 = cv2.Sobel(img2, cv2.CV_64F, 1, 0, ksize=3)
sobelYImag2 = cv2.Sobel(img2, cv2.CV_64F, 0, 1, ksize=3)

# Calcular la magnitud de los gradientes (combinación de sobel_x y sobel_y)
sobelCombined1 = cv2.magnitude(sobelXImag1, sobelYImag1)
sobelCombined2 = cv2.magnitude(sobelXImag2, sobelYImag2)

# Visualizar imágenes originales y procesadas
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
plt.imshow(img1,  cmap='gray')
plt.title('Imagen original 1')

plt.subplot(2, 2, 2)
plt.imshow(sobelCombined1, cmap='gray')
plt.title('Imagen procesada 1 con sobel')

plt.subplot(2, 2, 3)
plt.imshow(img2, cmap='gray')
plt.title('Imagen original 2')

plt.subplot(2, 2, 4)
plt.imshow(sobelCombined2, cmap='gray')
plt.title('Imagen procesada 2 con filtro de mediana')

plt.show()