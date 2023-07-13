import cv2
import numpy as np

def dibujar(mask, color):
    contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    objetos = 0  # Variable para contar objetos detectados
    for c in contornos:
        area = cv2.contourArea(c)
        if area > 3000:
            M = cv2.moments(c)
            if M["m00"] == 0:
                M["m00"] = 1
            x = int(M["m10"] / M["m00"])
            y = int(M['m01'] / M['m00'])
            nuevoContorno = cv2.convexHull(c)
            cv2.circle(frame, (x, y), 7, (0, 255, 0), -1)
            cv2.putText(frame, '{},{}'.format(x, y), (x + 10, y), font, 0.75, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.drawContours(frame, [nuevoContorno], 0, color, 3)
            
            # Check if object passes between the two lines with minimum separation
            if line_pos1 < y < line_pos2 and (x, y) not in objetos_detectados and abs(y - line_pos1) > min_separacion:
                objetos_detectados.add((x, y))
                if color == (255, 0, 0):
                    objetos_azul[0] += 1
                    total_objetos[0] += 1
                elif color == (0, 255, 255):
                    objetos_amarillo[0] += 1
                    total_objetos[0] += 1
                elif color == (0, 0, 255):
                    objetos_rojo[0] += 1
                    total_objetos[0] += 1
                elif color == (255, 255, 255):
                    objetos_branco[0] += 1
                    total_objetos[0] += 1
                objetos += 1
    return objetos

cap = cv2.VideoCapture(2)

azulBajo = np.array([100, 100, 20], np.uint8)
azulAlto = np.array([125, 255, 255], np.uint8)

amarilloBajo = np.array([15, 100, 20], np.uint8)
amarilloAlto = np.array([45, 255, 255], np.uint8)

redBajo1 = np.array([0, 100, 20], np.uint8)
redAlto1 = np.array([5, 255, 255], np.uint8)

redBajo2 = np.array([175, 100, 20], np.uint8)
redAlto2 = np.array([179, 255, 255], np.uint8)

brancoBajo = np.array([164,180,204], np.uint8)
brancoAlto = np.array([166, 182, 206], np.uint8)

font = cv2.FONT_HERSHEY_SIMPLEX

# Variables to store the counts
objetos_azul = [0]
objetos_amarillo = [0]
objetos_rojo = [0]
objetos_branco = [0]
total_objetos = [0]

# Line positions
line_pos1 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) / 3)
line_pos2 = int(2 * cap.get(cv2.CAP_PROP_FRAME_HEIGHT) / 5.7)

# Minimum separation between the lines
min_separacion = 0  # Adjust this value as needed

# Set to store detected objects
objetos_detectados = set()

while True:
    ret, frame = cap.read()

    if ret:
        
        frameHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        maskAzul = cv2.inRange(frameHSV, azulBajo, azulAlto)
        maskAmarillo = cv2.inRange(frameHSV, amarilloBajo, amarilloAlto)
        maskBranco = cv2.inRange(frameHSV, brancoBajo, brancoAlto)
        maskRed1 = cv2.inRange(frameHSV, redBajo1, redAlto1)
        maskRed2 = cv2.inRange(frameHSV, redBajo2, redAlto2)
        maskRed = cv2.add(maskRed1, maskRed2)

        objetos_azul[0] = dibujar(maskAzul, (255, 0, 0))
        objetos_amarillo[0] = dibujar(maskAmarillo, (0, 255, 255))
        objetos_rojo[0] = dibujar(maskRed, (0, 0, 255))
        objetos_branco[0] = dibujar(maskBranco, (255, 255, 255))

        cv2.putText(frame, 'Azul: {}'.format(objetos_azul[0]), (10, 20), font, 0.7, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, 'Amarelho: {}'.format(objetos_amarillo[0]), (10, 40), font, 0.7, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, 'Vermelho: {}'.format(objetos_rojo[0]), (10, 60), font, 0.7, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, 'Branco: {}'.format(objetos_branco[0]), (10, 80), font, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, 'Total: {}'.format(total_objetos[0]), (10, 100), font, 0.7, (0, 0, 0), 1, cv2.LINE_AA)

        cv2.line(frame, (0, line_pos1), (frame.shape[1], line_pos1), (0, 255, 0), 2)
        cv2.line(frame, (0, line_pos2), (frame.shape[1], line_pos2), (0, 255, 0), 2)
        #rotation = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('s'):
        break

cap.release()
cv2.destroyAllWindows()
