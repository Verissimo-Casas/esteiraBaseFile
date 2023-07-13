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
            if line1 < y < line2:
                count += 1

    return count

azulBajo = np.array([100,100,20],np.uint8)
azulAlto = np.array([125,255,255],np.uint8)

amarilloBajo = np.array([15,100,20],np.uint8)
amarilloAlto = np.array([45,255,255],np.uint8)

rojoBajo1 = np.array([0,100,20],np.uint8)
rojoAlto1 = np.array([5,255,255],np.uint8)

rojoBajo2 = np.array([175,100,20],np.uint8)
rojoAlto2 = np.array([179,255,255],np.uint8)

verdeBajo = np.array([35,100,20],np.uint8)
verdeAlto = np.array([85,255,255],np.uint8)

naranjaBajo = np.array([10,100,20],np.uint8)
naranjaAlto = np.array([25,255,255],np.uint8)

marronBajo = np.array([0,60,20],np.uint8)
marronAlto = np.array([15,255,255],np.uint8)

grisBajo = np.array([0,0,50],np.uint8)
grisAlto = np.array([179,50,255],np.uint8)

violetaBajo = np.array([125,50,50],np.uint8)
violetaAlto = np.array([155,255,255],np.uint8)

blancoBajo = np.array([0,0,200],np.uint8)
blancoAlto = np.array([179,30,255],np.uint8)

font = cv2.FONT_HERSHEY_SIMPLEX

cap = cv2.VideoCapture(0)

LINE1_Y = 200
LINE2_Y = 220

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
        maskRojo1 = cv2.inRange(frameHSV, rojoBajo1, rojoAlto1)
        maskRojo2 = cv2.inRange(frameHSV, rojoBajo2, rojoAlto2)
        maskRojo = cv2.add(maskRojo1, maskRojo2)
        maskVerde = cv2.inRange(frameHSV, verdeBajo, verdeAlto)
        maskNaranja = cv2.inRange(frameHSV, naranjaBajo, naranjaAlto)
        maskMarron = cv2.inRange(frameHSV, marronBajo, marronAlto)
        maskGris = cv2.inRange(frameHSV, grisBajo, grisAlto)
        maskVioleta = cv2.inRange(frameHSV, violetaBajo, violetaAlto)
        maskBlanco = cv2.inRange(frameHSV, blancoBajo, blancoAlto)

        count_azul += dibujar(maskAzul, (255, 0, 0), LINE1_Y, LINE2_Y)
        count_amarillo += dibujar(maskAmarillo, (0, 255, 255), LINE1_Y, LINE2_Y)
        count_rojo += dibujar(maskRojo, (0, 0, 255), LINE1_Y, LINE2_Y)
        count_verde += dibujar(maskVerde, (0, 255, 0), LINE1_Y, LINE2_Y)


        totalCnts = count_azul + count_amarillo + count_rojo + count_verde + count_naranja + count_marron + count_gris + count_violeta + count_blanco

        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        cv2.line(frame, (0, LINE1_Y), (frame.shape[1], LINE1_Y), (0, 0, 255), 1)
        cv2.line(frame, (0, LINE2_Y), (frame.shape[1], LINE2_Y), (0, 0, 255), 1)

        cv2.circle(frame, (30, 40), 15, (255, 0, 0), -1)
        cv2.circle(frame, (30, 80), 15, (0, 255, 255), -1)
        cv2.circle(frame, (30, 120), 15, (0, 0, 255), -1)
        cv2.circle(frame, (30, 160), 15, (0, 255, 0), -1)
        cv2.circle(frame, (30, 200), 15, (0, 165, 255), -1)
        cv2.circle(frame, (30, 240), 15, (42, 42, 165), -1)
        cv2.circle(frame, (30, 280), 15, (128, 128, 128), -1)
        cv2.circle(frame, (30, 320), 15, (238, 130, 238), -1)
        cv2.circle(frame, (30, 360), 15, (255, 255, 255), -1)

        cv2.putText(frame, str(count_azul), (65, 50), font, 0.75, (0, 0, 0), 1)
        cv2.putText(frame, str(count_amarillo), (65, 90), font, 0.75, (0, 0, 0), 1)
        cv2.putText(frame, str(count_rojo), (65, 130), font, 0.75, (0, 0, 0), 1)
        cv2.putText(frame, str(count_verde), (65, 170), font, 0.75, (0, 0, 0), 1)


        cv2.putText(frame, str(totalCnts), (55, 410), font, 0.75, (0, 0, 0), 1)

        cv2.imshow('frame', frame)
        if cv2.waitKey(70) & 0xFF == ord('s'):
            break

cap.release()
cv2.destroyAllWindows()
