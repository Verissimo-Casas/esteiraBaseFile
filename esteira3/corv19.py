import cv2
import numpy as np

def dibujar(mask, color, line1, line2):
    contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    count = 0
    for c in contornos:
        area = cv2.contourArea(c)
        if area > 1000:
            M = cv2.moments(c)
            if M["m00"] == 0:
                M["m00"] = 1
            x = int(M["m10"] / M["m00"])
            y = int(M['m01'] / M['m00'])
            nuevoContorno = cv2.convexHull(c)
            cv2.circle(frame, (x, y), 7, color, -1)
            cv2.putText(frame, '{},{}'.format(x, y), (x + 10, y), font, 0.75, color, 1, cv2.LINE_AA)
            cv2.drawContours(frame, [nuevoContorno], 0, color, 1)

            # Check if object passes between the two lines
            if line1 < x < line2:
                count += 1

    return count

azulBajo = np.array([100,100,20],np.uint8)
azulAlto = np.array([125,255,255],np.uint8)

amarilloBajo = np.array([15,100,20],np.uint8)
amarilloAlto = np.array([45,255,255],np.uint8)

redBajo1 = np.array([0,100,20],np.uint8)
redAlto1 = np.array([5,255,255],np.uint8)

redBajo2 = np.array([175,100,20],np.uint8)
redAlto2 = np.array([179,255,255],np.uint8)

font = cv2.FONT_HERSHEY_SIMPLEX

cap = cv2.VideoCapture(0)

count_azul = 0
count_amarillo = 0
count_rojo = 0
count_verde = 0
count_naranja = 0
count_marron = 0
count_gris = 0
count_violeta = 0
count_blanco = 0
totalCnts = 0

while True:
    ret, frame = cap.read()

    if ret == True:
        frameHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        maskAzul = cv2.inRange(frameHSV, azulBajo, azulAlto)
        maskAmarillo = cv2.inRange(frameHSV, amarilloBajo, amarilloAlto)
        maskRojo1 = cv2.inRange(frameHSV, redBajo1, redAlto1)
        maskRojo2 = cv2.inRange(frameHSV, redBajo2, redAlto2)

        maskRojo = cv2.add(maskRojo1, maskRojo2)

        # drawing lines
        LINE1_Y = 500
        LINE2_Y = 520
        cv2.line(frame, (LINE1_Y, 10), (LINE1_Y, frame.shape[1]), (0, 0, 255), 1)
        cv2.line(frame, (LINE2_Y, 10), (LINE2_Y, frame.shape[1]), (0, 0, 255), 1)

        count_azul += dibujar(maskAzul, (255, 0, 0), LINE1_Y, LINE2_Y)
        count_amarillo += dibujar(maskAmarillo, (0, 255, 255), LINE1_Y, LINE2_Y)
        count_rojo += dibujar(maskRojo, (0, 0, 255), LINE1_Y, LINE2_Y)

        totalCnts = count_azul + count_amarillo + count_rojo

        # drwaing circles
        cv2.circle(frame, (30, 40), 15, (255, 0, 0), -1)
        cv2.circle(frame, (30, 80), 15, (0, 255, 255), -1)
        cv2.circle(frame, (30, 120), 15, (0, 0, 255), -1)


        cv2.putText(frame, f"Azul: {str(count_azul)}", (65, 50), font, 1, (255, 0, 0), 2)
        cv2.putText(frame, f"Amarelo: {str(count_amarillo)}", (65, 90), font, 1, (45,255,255), 2)
        cv2.putText(frame, f"Vermelho: {str(count_rojo)}", (65, 130), font, 1, (0, 0, 255), 2)

        cv2.putText(frame, f"Total = {str(totalCnts)}", (55, 410), font, 1, (0, 0, 0), 3)

        cv2.imshow('frame', frame)
        if cv2.waitKey(70) & 0xFF == ord('s'):
            break

cap.release()
cv2.destroyAllWindows()
