import cv2
import numpy as np

def dibujar(mask, color):
    contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    objetos = 0  # Variable para contar objetos detectados
    for c in contornos:
        area = cv2.contourArea(c)
        if area > 3000:
            objetos += 1  # Incrementar el contador de objetos detectados
            M = cv2.moments(c)
            if M["m00"] == 0:
                M["m00"] = 1
            x = int(M["m10"] / M["m00"])
            y = int(M['m01'] / M['m00'])
            nuevoContorno = cv2.convexHull(c)
            cv2.circle(frame, (x, y), 7, (0, 255, 0), -1)
            cv2.putText(frame, '{},{}'.format(x, y), (x + 10, y), font, 0.75, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.drawContours(frame, [nuevoContorno], 0, color, 3)
    return objetos

cap = cv2.VideoCapture(0)

azulBajo = np.array([100, 100, 20], np.uint8)
azulAlto = np.array([125, 255, 255], np.uint8)

amarilloBajo = np.array([15, 100, 20], np.uint8)
amarilloAlto = np.array([45, 255, 255], np.uint8)

redBajo1 = np.array([0, 100, 20], np.uint8)
redAlto1 = np.array([5, 255, 255], np.uint8)

redBajo2 = np.array([175, 100, 20], np.uint8)
redAlto2 = np.array([179, 255, 255], np.uint8)

font = cv2.FONT_HERSHEY_SIMPLEX
while True:
    ret, frame = cap.read()

    if ret:
        frameHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        maskAzul = cv2.inRange(frameHSV, azulBajo, azulAlto)
        maskAmarillo = cv2.inRange(frameHSV, amarilloBajo, amarilloAlto)
        maskRed1 = cv2.inRange(frameHSV, redBajo1, redAlto1)
        maskRed2 = cv2.inRange(frameHSV, redBajo2, redAlto2)
        maskRed = cv2.add(maskRed1, maskRed2)

        objetos_azul = dibujar(maskAzul, (255, 0, 0))
        objetos_amarillo = dibujar(maskAmarillo, (0, 255, 255))
        objetos_rojo = dibujar(maskRed, (0, 0, 255))
        total_objetos = objetos_azul + objetos_amarillo + objetos_rojo

        cv2.putText(frame, 'Azul: {}'.format(objetos_azul), (10, 20), font, 0.7, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, 'Amarillo: {}'.format(objetos_amarillo), (10, 40), font, 0.7, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, 'Rojo: {}'.format(objetos_rojo), (10, 60), font, 0.7, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, 'Total: {}'.format(total_objetos), (10, 80), font, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow('frame', frame)


    if cv2.waitKey(1) & 0xFF == ord('s'):
        break

cap.release()
cv2.destroyAllWindows()
