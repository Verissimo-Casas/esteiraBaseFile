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
line_separation = 30

# Función para calcular el centroide de un contorno
def calculate_centroid(contour):
    M = cv2.moments(contour)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    return (cX, cY)

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

            # Calcular y almacenar los centroides
            for contour in contours:
                centroid = calculate_centroid(contour)
                centroids.append((centroid, color))

    # Verificar si los centroides están entre las líneas de detección y contarlos
    for centroid, color in centroids:
        if line_separation <= centroid[1] <= (line_separation * 2):
            object_counters[color] += 1
            total_counter += 1
            # Dibujar el centroide del objeto
            cv2.circle(frame, centroid, 5, (0, 0, 255), -1)
            # Colocar etiqueta de color cerca del contorno
            cv2.putText(frame, color, (centroid[0] + 10, centroid[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    centroids.clear()

    # Dibujar la línea en el centro
    line_y = line_separation + int(line_separation / 2)
    cv2.line(frame, (0, line_y), (frame.shape[1], line_y), (255, 0, 0), 2)

    # Mostrar los resultados en la imagen
    cv2.putText(frame, 'Total: {}'.format(total_counter), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    y = 60
    for color, count in object_counters.items():
        cv2.putText(frame, '{}: {}'.format(color, count), (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        y += 30

    return frame

# Función para dibujar las líneas de detección
def draw_detection_lines(frame, y_top, y_bottom):
    cv2.line(frame, (0, y_top), (frame.shape[1], y_top), (0, 0, 255), 2)
    cv2.line(frame, (0, y_bottom), (frame.shape[1], y_bottom), (0, 0, 255), 2)


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

        # Dibujar las líneas de detección
        draw_detection_lines(frame, line_separation, line_separation * 2)

        # Detectar y contar objetos
        result = detect_and_count_objects(frame)

        # Mostrar el marco resultante
        cv2.imshow('Object Detection', result)

        # Salir del bucle al presionar la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar los recursos
    cap.release()
    cv2.destroyAllWindows()

# Ejecutar la función principal
if __name__ == '__main__':
    main()
