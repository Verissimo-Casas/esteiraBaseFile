import cv2
import numpy as np

# Cargar la imagen
image = cv2.imread('detection-opencv/figurasColores.jpg')

# Definir el rango de color azul en el espacio BGR
azulBajo = np.array([100, 0, 0], np.uint8)
azulAlto = np.array([255, 50, 50], np.uint8)

# Convertir la imagen a espacio de color HSV
imageHSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Crear la máscara de color azul
maskAzul = cv2.inRange(imageHSV, azulBajo, azulAlto)

# Aplicar la máscara a la imagen original
result = cv2.bitwise_and(image, image, mask=maskAzul)

# Mostrar la imagen original y la imagen con la máscara aplicada
cv2.imshow('Original', image)
cv2.imshow('Mask Azul', maskAzul)
cv2.imshow('Result', result)

# Esperar a que se presione una tecla y luego cerrar las ventanas
cv2.waitKey(0)
cv2.destroyAllWindows()
