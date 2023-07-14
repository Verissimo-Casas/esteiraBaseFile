import cv2
import numpy as np

def find_contours(mask):
    return cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

def calculate_centroid(contour):
    moments = cv2.moments(contour)
    if moments["m00"] == 0:
        moments["m00"] = 1
    x = int(moments["m10"] / moments["m00"])
    y = int(moments["m01"] / moments["m00"])
    return x, y

def draw_circle_and_text(frame, x, y, color):
    cv2.circle(frame, (x, y), 7, color, -1)
    cv2.putText(frame, '{},{}'.format(x, y), (x + 10, y + 10), font, 0.75, color, 1, cv2.LINE_AA)

def draw_contour(frame, contour, color):
    nuevo_contorno = cv2.convexHull(contour)
    cv2.drawContours(frame, [nuevo_contorno], 0, color, 1)

def dibujar(mask, color):
    contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contornos:
        area = cv2.contourArea(c)
        if area > 3000:
            x, y = calculate_centroid(c)
            draw_circle_and_text(frame, x, y)
            draw_contour(frame, c, color)

def dibujar(mask, color):
    contornos = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    for c in contornos:
        area = cv2.contourArea(c)
        if area > 3000:
            moments = cv2.moments(c)
            if moments["m00"] == 0:
                moments["m00"] = 1
            x = int(moments["m10"] / moments["m00"])
            y = int(moments["m01"] / moments["m00"])
            cv2.circle(frame, (x, y), 7, (0, 255, 0), -1)
            cv2.putText(frame, '{},{}'.format(x, y), (x + 10, y), font, 0.75, (0, 255, 0), 1, cv2.LINE_AA)
            nuevo_contorno = cv2.convexHull(c)
            cv2.drawContours(frame, [nuevo_contorno], 0, color, 3)




azulBajo = np.array([100,100,20],np.uint8)
azulAlto = np.array([125,255,255],np.uint8)

amarilloBajo = np.array([15,100,20],np.uint8)
amarilloAlto = np.array([45,255,255],np.uint8)

redBajo1 = np.array([0,100,20],np.uint8)
redAlto1 = np.array([5,255,255],np.uint8)

redBajo2 = np.array([175,100,20],np.uint8)
redAlto2 = np.array([179,255,255],np.uint8)

font = cv2.FONT_HERSHEY_SIMPLEX

cap = cv2.VideoCapture(2)

while True:

    ret,frame = cap.read()

    if not ret:
        break

    frameHSV = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

    maskAzul = cv2.inRange(frameHSV,azulBajo,azulAlto)
    maskAmarillo = cv2.inRange(frameHSV,amarilloBajo,amarilloAlto)
    maskRed1 = cv2.inRange(frameHSV,redBajo1,redAlto1)
    maskRed2 = cv2.inRange(frameHSV,redBajo2,redAlto2)

    maskRed = cv2.add(maskRed1,maskRed2)

    dibujar(maskAzul, (255, 0, 0))
    dibujar(maskAmarillo, (0, 255, 255))
    dibujar(maskRed, (0, 0, 255))

    cv2.imshow('frame',frame)
    cv2.imshow('maskAzul',maskAzul)
    cv2.imshow('maskAmarillo',maskAmarillo)
    cv2.imshow('maskRed',maskRed)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()