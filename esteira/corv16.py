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

# Variables para el seguimiento del centroide
centroids = []

# Separación mínima entre las líneas de detección
line_separation = 15

# Umbral mínimo de área para considerar un contorno como objeto válido
min_area_threshold = 2000

# Función para calcular el centroide de un contorno
def calculate_centroid(contour):
    M = cv2.moments(contour)
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
    global total_counter, centroids

    # Convertir la imagen al espacio de color HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Iterar sobre los rangos de color
    for color, (lower, upper) in color_ranges.items():
        # Crear una máscara para el rango de color actual
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))

        # Aplicar operaciones morfológicas para eliminar ruido
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        # Encontrar contornos de objetos en la máscara
        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Verificar si se encontraron contornos
        if len(contours) > 0:
            # Dibujar los contornos encontrados en el marco original
            cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)

            # Calcular y almacenar los centroides de los objetos válidos
            for contour in contours:
                # Calcular el área del contorno
                area = cv2.contourArea(contour)
                if area > min_area_threshold:
                    centroid = calculate_centroid(contour)
                    centroids.append((centroid, color))

    # Verificar si los centroides pasan entre las líneas de detección y contarlos
    for centroid, color in centroids:
        line_y_top = frame.shape[0] // 2 - line_separation // 2
        line_y_bottom = frame.shape[0] // 2 + line_separation // 2
        if line_y_top <= centroid[1] <= line_y_bottom:
            object_counters[color] += 1
            total_counter += 1

    # Dibujar los centroides de los objetos
    for centroid, color in centroids:
        cv2.circle(frame, centroid, 5, (0, 0, 255), -1)
        cv2.putText(frame, color, (centroid[0] + 10, centroid[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    centroids.clear()

    # Calcular la posición vertical del centro de las líneas de detección
    line_y_center = frame.shape[0] // 2

    # Dibujar las líneas de detección
    draw_detection_lines(frame, line_y_center)

    # Mostrar los resultados en la imagen
    cv2.putText(frame, 'Total: {}'.format(total_counter), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    y = 60
    for color, count in object_counters.items():
        # Obtener el color en formato BGR para mostrarlo correctamente
        bgr_color = np.uint8([[color_ranges[color][0]]])
        bgr_color = cv2.cvtColor(bgr_color, cv2.COLOR_HSV2BGR)
        bgr_color = tuple(bgr_color[0][0].tolist())

        cv2.putText(frame, '{}: {}'.format(color, count), (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, bgr_color, 2)
        y += 30

    return frame

# Función principal
def main():
    # Capturar video desde la cámara
    cap = cv2.VideoCapture(0)

    # Establecer la resolución de la captura a 720p
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while True:
        # Leer el siguiente marco del video
        ret, frame = cap.read()
        if not ret:
            break

        # Invertir el marco horizontalmente para que coincida con la vista en espejo de la cámara
        frame = cv2.flip(frame, 1)

        # Redimensionar la imagen a una resolución de 720p
        frame = cv2.resize(frame, (1280, 720))

        # Detectar y contar objetos
        result = detect_and_count_objects(frame)

        # Mostrar el marco resultante
        cv2.imshow('Objeto Detector', result)

        # Salir del bucle si se presiona la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar los recursos
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
