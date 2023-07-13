import cv2
import numpy as np
from datetime import datetime

# Definir rangos de color en HSV para cada objeto
color_ranges = {
    "azul": ([100, 50, 50], [130, 255, 255]),
    "rojo": ([0, 50, 50], [10, 255, 255]),
    "verde": ([40, 50, 50], [80, 255, 255]),
    "amarillo": ([20, 50, 50], [40, 255, 255])
}

# Mapa de colores para dibujar los contornos y el texto
color_map = {
    "azul": (255, 0, 0),
    "rojo": (0, 0, 255),
    "verde": (0, 255, 0),
    "amarillo": (0, 255, 255)
}

# Inicializar el acumulado de objetos por color
cantidad_objetos = {
    "azul": 0,
    "rojo": 0,
    "verde": 0,
    "amarillo": 0
}

# Inicializar el tiempo de inicio y el número de cuadros procesados
start_time = datetime.now()
frame_count = 0

# Capturar video desde la cámara
cap = cv2.VideoCapture(2)

while True:
    # Leer un cuadro del video
    ret, frame = cap.read()
    if not ret:
        break

    # Incrementar el contador de cuadros
    frame_count += 1

    # Convertir el cuadro a espacio de color HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Procesar cada rango de color
    for color, (lower, upper) in color_ranges.items():
        # Crear una máscara para el rango de color
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))

        # Aplicar operaciones morfológicas para eliminar el ruido
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Encontrar los contornos de los objetos en la máscara
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filtrar los contornos por área
        filtered_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if 1000 < area < 5000:
                filtered_contours.append(contour)

        # Dibujar los contornos y calcular los centroides de cada objeto
        for contour in filtered_contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                x = int(M["m10"] / M["m00"])
                y = int(M["m01"] / M["m00"])
                cv2.drawContours(frame, [contour], 0, color_map[color], 2)
                cv2.circle(frame, (x, y), 7, color_map[color], -1)
                cv2.putText(frame, color, (x - 20, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_map[color], 2)
                cantidad_objetos[color] += 1

    # Calcular el FPS (cuadros por segundo)
    end_time = datetime.now()
    elapsed_time = (end_time - start_time).total_seconds()
    fps = frame_count / elapsed_time
    cv2.putText(frame, f"FPS: {fps:.2f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Calcular el acumulado por color y el acumulado total
    cantidad_total = sum(cantidad_objetos.values())
    for color, cantidad in cantidad_objetos.items():
        cv2.putText(frame, f"{color}: {cantidad}", (20, 80 + 30 * int(color_map[color][1] > 0)), cv2.FONT_HERSHEY_SIMPLEX, 1, color_map[color], 2)
    cv2.putText(frame, f"Total: {cantidad_total}", (20, 80 + 30 * len(cantidad_objetos)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Mostrar el cuadro en una ventana
    cv2.imshow("Video", frame)

    # Salir del bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Liberar los recursos
cap.release()
cv2.destroyAllWindows()
