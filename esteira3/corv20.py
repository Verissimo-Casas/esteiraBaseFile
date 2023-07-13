import cv2
import numpy as np

def dibujar(mask, color, line1, line2):
    contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    count = 0
    for c in contornos:
        area = cv2.contourArea(c)
        if area > 3000:
            M = cv2.moments(c)
            if M["m00"] == 0:
                M["m00"] = 1
            x = int(M["m10"] / M["m00"])
            y = int(M['m01'] / M['m00'])
            nuevoContorno = cv2.convexHull(c)
            cv2.circle(frame, (x, y), 7, color, -1)
            cv2.putText(frame, '{},{}'.format(x, y), (x + 10, y), font, 0.75, color, 1, cv2.LINE_AA)
            cv2.drawContours(frame, [nuevoContorno], 0, color, 3)

            # Check if object passes between the two lines
            if line1 < y < line2:
                count += 1

    return count

azulBajo = np.array([100,100,20],np.uint8)
azulAlto = np.array([125,255,255],np.uint8)

amarilloBajo = np.array([15,100,20],np.uint8)
amarilloAlto = np.array([45,255,255],np.uint8)

font = cv2.FONT_HERSHEY_SIMPLEX

cap = cv2.VideoCapture(0)

# Set line positions
LINE1_Y = 200
LINE2_Y = 210

count_azul = 0
count_amarillo = 0
totalCnts = 0

while True:
    ret, frame = cap.read()

    if ret == True:
        frameHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        maskAzul = cv2.inRange(frameHSV, azulBajo, azulAlto)
        maskAmarillo = cv2.inRange(frameHSV, amarilloBajo, amarilloAlto)

        count_azul += dibujar(maskAzul, (255, 0, 0), LINE1_Y, LINE2_Y)
        count_amarillo += dibujar(maskAmarillo, (0, 255, 255), LINE1_Y, LINE2_Y)

        totalCnts = count_azul + count_amarillo

        cv2.line(frame, (0, LINE1_Y), (frame.shape[1], LINE1_Y), (0, 0, 255), 2)
        cv2.line(frame, (0, LINE2_Y), (frame.shape[1], LINE2_Y), (0, 0, 255), 2)

        cv2.putText(frame, str(count_azul), (65, 50), font, 0.75, (0, 0, 0), 1)
        cv2.putText(frame, str(count_amarillo), (65, 90), font, 0.75, (0, 0, 0), 1)

        cv2.putText(frame, str(totalCnts), (55, 410), font, 0.75, (0, 0, 0), 1)

        cv2.imshow('frame', frame)
        if cv2.waitKey(70) & 0xFF == ord('s'):
            break

cap.release()
cv2.destroyAllWindows()
