import numpy as np
import cv2

# Variables globales para los rangos de color
rango_azul_bajo = np.array([0, 0, 0], np.uint8)
rango_azul_alto = np.array([0, 0, 0], np.uint8)
rango_rojo_bajo = np.array([0, 0, 0], np.uint8)
rango_rojo_alto = np.array([0, 0, 0], np.uint8)
rango_verde_bajo = np.array([0, 0, 0], np.uint8)
rango_verde_alto = np.array([0, 0, 0], np.uint8)
rango_amarillo_bajo = np.array([0, 0, 0], np.uint8)
rango_amarillo_alto = np.array([0, 0, 0], np.uint8)

def definir_rango_color(event, x, y, flags, param):
    global rango_azul_bajo, rango_azul_alto, rango_rojo_bajo, rango_rojo_alto, rango_verde_bajo, rango_verde_alto, rango_amarillo_bajo, rango_amarillo_alto
    if event == cv2.EVENT_LBUTTONDOWN:
        pixel = frameHSV[y, x]
        color = frame[y, x]
        rango_bajo = np.array([pixel[0] - 10, pixel[1] - 50, pixel[2] - 50], np.uint8)
        rango_alto = np.array([pixel[0] + 10, pixel[1] + 50, pixel[2] + 50], np.uint8)
        rango_azul_bajo = rango_bajo
        rango_azul_alto = rango_alto
        # Actualizar los otros rangos de color según sea necesario
        print("Rango azul bajo:", rango_azul_bajo)
        print("Rango azul alto:", rango_azul_alto)
        
        print("Rango rojo bajo:", rango_rojo_bajo)
        print("Rango rojo alto:", rango_rojo_alto)

        print("Rango verde bajo:", rango_verde_bajo)
        print("Rango verde alto:", rango_verde_alto)

        print("Rango amarillo bajo:", rango_amarillo_bajo)
        print("Rango amarillo alto:", rango_amarillo_alto)


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

    # Aplicar la máscara al frame
    resul = cv2.bitwise_and(frame, frame, mask=mask_azul)

    # Mostrar el frame original y el resultado
    cv2.imshow('Video', np.hstack([frame, resul]))

    # Salir del bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la captura de video y cerrar las ventanas
cap.release()
cv2.destroyAllWindows()
