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

# Función para detectar y contar objetos
def detect_and_count_objects(frame):
    global total_counter

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

            # Incrementar el contador de objetos para el color actual
            object_counters[color] += len(contours)
            total_counter += len(contours)

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

    while True:
        # Leer el siguiente marco del video
        ret, frame = cap.read()
        if not ret:
            break

        # Invertir el marco horizontalmente para que coincida con la vista en espejo de la cámara
        frame = cv2.flip(frame, 1)

        # Dibujar las líneas de detección
        draw_detection_lines(frame, 200, 400)

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
