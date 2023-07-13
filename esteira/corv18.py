import cv2
import numpy as np

# Definir los rangos de color para cada objeto
color_ranges = {
    'rojo': ([0, 100, 100], [10, 255, 255]),  # Rango bajo y alto para el rojo
    'amarillo': ([20, 100, 100], [30, 255, 255]),  # Rango bajo y alto para el amarillo
    'azul': ([90, 100, 100], [120, 255, 255]),  # Rango bajo y alto para el azul
    'verde': ([45, 100, 100], [75, 255, 255]),  # Rango bajo y alto para el verde
    'blanco': ([0, 0, 200], [255, 30, 255]),  # Rango bajo y alto para el blanco
    'rosado': ([140, 100, 100], [170, 255, 255]),  # Rango bajo y alto para el rosado
    'violeta': ([130, 100, 100], [140, 255, 255]),  # Rango bajo y alto para el violeta
    'marron': ([10, 100, 100], [20, 255, 255]),  # Rango bajo y alto para el marrón
    'gris': ([0, 0, 100], [255, 30, 200])  # Rango bajo y alto para el gris
}

# Inicializar contadores de objetos y resultados acumulados
object_counters = {color: 0 for color in color_ranges}
total_counter = 0

# Variables para el seguimiento del centroide y etiqueta
centroids = []
labels = []

# Separación mínima entre las líneas de detección
line_separation = 8

# Umbral mínimo de área para considerar un contorno como objeto válido
min_area_threshold = 500

# Función para calcular el centroide de un contorno
def calculate_centroid(contour):
    M = cv2.moments(contour)
    if M["m00"] == 0:
        M["m00"] = 1
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    return (cX, cY)

# Función para dibujar las líneas de detección
def draw_detection_lines(frame, y_center):
    line_y_top = y_center - line_separation // 2
    line_y_bottom = y_center + line_separation // 2
    cv2.line(frame, (0, line_y_top), (frame.shape[1], line_y_top), (0, 0, 255), 2)
    cv2.line(frame, (0, line_y_bottom), (frame.shape[1], line_y_bottom), (0, 0, 255), 2)

# Función para detectar y contar objetos
def detect_and_count_objects(frame):
    global total_counter, centroids, labels

    # Convertir la imagen al espacio de color HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Iterar sobre los rangos de color
    for color, (lower, upper) in color_ranges.items():
        # Crear una máscara para el rango de color actual
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))

        # Aplicar operación de apertura para eliminar ruido
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Encontrar contornos de objetos en la máscara
        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Verificar si se encontraron contornos
        if len(contours) > 0:
            # Calcular y almacenar los centroides de los objetos válidos
            for contour in contours:
                # Aproximación de polígonos
                epsilon = 0.01 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)

                # Filtrado por área
                if cv2.contourArea(approx) > min_area_threshold:
                    centroid = calculate_centroid(approx)
                    centroids.append(centroid)
                    labels.append(color)

    # Verificar si los centroides pasan entre las líneas de detección y contarlos
    for centroid, color in zip(centroids, labels):
        line_y_top = frame.shape[0] // 2 - line_separation // 2
        line_y_bottom = frame.shape[0] // 2 + line_separation // 2
        if line_y_top <= centroid[1] <= line_y_bottom:
            object_counters[color] += 1
            total_counter += 1

    # Dibujar los contornos y centroides en el marco
    for contour, color in zip(contours, labels):
        cv2.drawContours(frame, [contour], 0, color_ranges[color][0], 2)
    
    for centroid, color in zip(centroids, labels):
        cv2.circle(frame, centroid, 5, color_ranges[color][0], -1)
        cv2.putText(frame, color, (centroid[0] + 10, centroid[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_ranges[color][0], 2)

    centroids.clear()
    labels.clear()

    # Calcular la posición vertical del centro de las líneas de detección
    line_y_center = frame.shape[0] // 2

    # Dibujar las líneas de detección
    draw_detection_lines(frame, line_y_center)

    # Mostrar los resultados en la imagen
    cv2.putText(frame, 'Total: {}'.format(total_counter), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Mostrar los resultados para cada color
    y = 60
    for color, count in object_counters.items():
        cv2.putText(frame, '{}: {}'.format(color, count), (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_ranges[color][0], 2)
        y += 30

    return frame

# Capturar el video de la cámara
cap = cv2.VideoCapture(0)

while True:
    # Leer el marco actual
    ret, frame = cap.read()

    # Verificar si se pudo capturar el marco
    if not ret:
        break

    # Invertir el marco horizontalmente para que coincida con la vista en espejo de la cámara
    frame = cv2.flip(frame, 1)

    # Detectar y contar los objetos en el marco
    frame = detect_and_count_objects(frame)

    # Mostrar el marco resultante
    cv2.imshow('Object Detection', frame)

    # Salir del bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) == ord('q'):
        break

# Liberar los recursos
cap.release()
cv2.destroyAllWindows()
