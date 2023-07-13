import numpy as np
import cv2

# Variables globales para los rangos de color
rango_azul_bajo = np.array([0, 0, 0], np.uint8)
rango_azul_alto = np.array([0, 0, 0], np.uint8)

def definir_rango_color(event, x, y, flags, param):
    global rango_azul_bajo, rango_azul_alto
    if event == cv2.EVENT_LBUTTONDOWN:
        pixel = frameHSV[y, x]
        rango_bajo = np.array([pixel[0] - 10, pixel[1] - 50, pixel[2] - 50], np.uint8)
        rango_alto = np.array([pixel[0] + 10, pixel[1] + 50, pixel[2] + 50], np.uint8)
        rango_azul_bajo = rango_bajo
        rango_azul_alto = rango_alto

# Captura de video
cap = cv2.VideoCapture(2)

# Crear una ventana de visualización
cv2.namedWindow('Video')

# Establecer el callback del mouse
cv2.setMouseCallback('Video', definir_rango_color)

while True:
    # Capturar frame
    ret, frame = cap.read()

    # Verificar si el frame se capturó correctamente
    if not ret:
        break

    # Convertir el frame a espacio de color HSV
    frameHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Crear la máscara utilizando el rango de color azul
    mask_azul = cv2.inRange(frameHSV, rango_azul_bajo, rango_azul_alto)

    # Encontrar contornos en la máscara
    contours, _ = cv2.findContours(mask_azul, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Dibujar bounding boxes alrededor de los contornos
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Mostrar el frame original
    cv2.imshow('Video', frame)

    # Salir del bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la captura de video y cerrar las ventanas
cap.release()
cv2.destroyAllWindows()
