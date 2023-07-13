import cv2
import numpy as np

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        pixel = imageHSV[y, x]
        azulBajo = np.array([pixel[0]-10, 100, 100], np.uint8)
        azulAlto = np.array([pixel[0]+10, 255, 255], np.uint8)
        maskAzul = cv2.inRange(imageHSV, azulBajo, azulAlto)
        result = cv2.bitwise_and(image, image, mask=maskAzul)
        cv2.imshow('Mask Azul', maskAzul)
        cv2.imshow('Result', result)

# Cargar la imagen
image = cv2.imread('detection-opencv/figurasColores.jpg')

# Convertir la imagen a espacio de color HSV
imageHSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Mostrar la imagen original
cv2.imshow('Original', image)

# Configurar el callback del mouse para obtener los límites de color azul
cv2.setMouseCallback('Original', mouse_callback)

# Esperar a que se presione 'q' para cerrar la ventana
while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cerrar las ventanas
cv2.destroyAllWindows()
