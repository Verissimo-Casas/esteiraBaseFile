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

def draw_circle_and_text(frame, x, y):
    cv2.circle(frame, (x, y), 7, (0, 255, 0), -1)
    cv2.putText(frame, '{},{}'.format(x, y), (x + 10, y), font, 0.75, (0, 255, 0), 1, cv2.LINE_AA)

def draw_contour(frame, contour, color):
    nuevo_contorno = cv2.convexHull(contour)
    cv2.drawContours(frame, [nuevo_contorno], 0, color, 3)

def dibujar(mask, color):
    contornos = find_contours(mask)
    for c in contornos:
        area = cv2.contourArea(c)
        if area > 3000:
            x, y = calculate_centroid(c)
            draw_circle_and_text(frame, x, y)
            draw_contour(frame, c, color)



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
while True:

  ret,frame = cap.read()

  if ret == True:
    frameHSV = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    maskAzul = cv2.inRange(frameHSV,azulBajo,azulAlto)
    maskAmarillo = cv2.inRange(frameHSV,amarilloBajo,amarilloAlto)
    maskRed1 = cv2.inRange(frameHSV,redBajo1,redAlto1)
    maskRed2 = cv2.inRange(frameHSV,redBajo2,redAlto2)
    maskRed = cv2.add(maskRed1,maskRed2)

    # Opencv 4
    dibujar(maskAzul,(255,0,0))
    dibujar(maskAmarillo,(0,255,255))
    dibujar(maskRed,(0,0,255))

    #Imagen Resumen
    imgResumen = 255 * np.ones((210,100,3), dtype = np.uint8)
    cv2.circle(imgResumen, (30,30), 15, (255,0,0), -1)
    cv2.circle(imgResumen, (30,70), 15, (0,255,255), -1)
    cv2.circle(imgResumen, (30,110), 15, (0,0,255), -1)

    
    cv2.putText(imgResumen,str(len(maskAzul)),(65,40), 1, 2,(0,0,0),1)
    cv2.putText(imgResumen,str(len(maskAmarillo)),(65,80), 1, 2,(0,0,0),1)
    cv2.putText(imgResumen,str(len(maskRed)),(65,120), 1, 2,(0,0,0),1)

    totalCnts = len(maskAzul) + len(maskAmarillo) + len(maskRed)
    cv2.putText(imgResumen,str(totalCnts),(55,200), 1, 2,(0,0,0),1)
    cv2.imshow('Resumen', imgResumen)
    

    cv2.imshow('frame',frame)
    if cv2.waitKey(70) & 0xFF == ord('s'):
      break
cap.release()
cv2.destroyAllWindows()