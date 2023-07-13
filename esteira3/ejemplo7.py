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

color_map = {
    "azul": (255, 0, 0),
    "rojo": (0, 0, 255),
    "verde": (0, 255, 0),
    "amarillo": (0, 255, 255)
}

def definir_rango_color(event, x, y, flags, param):
    global rango_azul_bajo, rango_azul_alto, rango_rojo_bajo, rango_rojo_alto, rango_verde_bajo, rango_verde_alto, rango_amarillo_bajo, rango_amarillo_alto
    if event == cv2.EVENT_LBUTTONDOWN:
        pixel = frameHSV[y, x]
        rango_bajo = np.array([pixel[0] - 10, pixel[1] - 50, pixel[2] - 50], np.uint8)
        rango_alto = np.array([pixel[0] + 10, pixel[1] + 50, pixel[2] + 50], np.uint8)
        if y < frame.shape[0] // 2:
            rango_azul_bajo = rango_bajo
            rango_azul_alto = rango_alto
        else:
            if x < frame.shape[1] // 2:
                rango_rojo_bajo = rango_bajo
                rango_rojo_alto = rango_alto
            else:
                rango_verde_bajo = rango_bajo
                rango_verde_alto = rango_alto
                rango_amarillo_bajo = rango_bajo
                rango_amarillo_alto = rango_alto

# Captura de video
cap = cv2.VideoCapture(2)

# Crear una ventana de visualización
cv2.namedWindow('Video')

# Establecer el callback del mouse
cv2.setMouseCallback('Video', definir_rango_color)

# Variables para el seguimiento del objeto
centroide = None
objeto_encontrado = False

while True:
    # Capturar frame
    ret, frame = cap.read()

    # Verificar si el frame se capturó correctamente
    if not ret:
        break

    # Convertir el frame a espacio de color HSV
    frameHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Crear las máscaras utilizando los rangos de color
    mask_azul = cv2.inRange(frameHSV, rango_azul_bajo, rango_azul_alto)
    mask_rojo = cv2.inRange(frameHSV, rango_rojo_bajo, rango_rojo_alto)
    mask_verde = cv2.inRange(frameHSV, rango_verde_bajo, rango_verde_alto)
    mask_amarillo = cv2.inRange(frameHSV, rango_amarillo_bajo, rango_amarillo_alto)

    # Encontrar contornos en las máscaras
    contours_azul, _ = cv2.findContours(mask_azul, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_rojo, _ = cv2.findContours(mask_rojo, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_verde, _ = cv2.findContours(mask_verde, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_amarillo, _ = cv2.findContours(mask_amarillo, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Definir umbral mínimo de área para filtrar los contornos pequeños
    area_minima = 100

    # Filtrar los contornos por área mínima
    contours_azul_filtrados = [contour for contour in contours_azul if cv2.contourArea(contour) > area_minima]
    contours_rojo_filtrados = [contour for contour in contours_rojo if cv2.contourArea(contour) > area_minima]
    contours_verde_filtrados = [contour for contour in contours_verde if cv2.contourArea(contour) > area_minima]
    contours_amarillo_filtrados = [contour for contour in contours_amarillo if cv2.contourArea(contour) > area_minima]

    # Dibujar contornos filtrados en la imagen original
    frame_with_contours = frame.copy()
    for contour in contours_azul_filtrados:
        cv2.drawContours(frame_with_contours, [contour], -1, color_map["azul"], 2)
    for contour in contours_rojo_filtrados:
        cv2.drawContours(frame_with_contours, [contour], -1, color_map["rojo"], 2)
    for contour in contours_verde_filtrados:
        cv2.drawContours(frame_with_contours, [contour], -1, color_map["verde"], 2)
    for contour in contours_amarillo_filtrados:
        cv2.drawContours(frame_with_contours, [contour], -1, color_map["amarillo"], 2)

    # Realizar seguimiento del objeto si se encuentra un solo contorno
    if len(contours_verde_filtrados) == 1:
        objeto_encontrado = True
        # Obtener el centroide del contorno
        M = cv2.moments(contours_verde_filtrados[0])
        if M["m00"] != 0:
            centroide_x = int(M["m10"] / M["m00"])
            centroide_y = int(M["m01"] / M["m00"])
            centroide = (centroide_x, centroide_y)
            # Dibujar el centroide en el frame
            cv2.circle(frame_with_contours, centroide, 5, color_map["verde"], -1)
    else:
        objeto_encontrado = False
        centroide = None

    # Mostrar el frame original y el frame con contornos
    cv2.imshow('Video', np.hstack((frame, frame_with_contours)))

    # Salir del bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la captura de video y cerrar las ventanas
cap.release()
cv2.destroyAllWindows()
