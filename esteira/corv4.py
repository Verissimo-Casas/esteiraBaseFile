import cv2
import numpy as np

capture = cv2.VideoCapture(0)
while True:
    ret, frame = capture.read()
    if ret == True:
        
        hvs = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        azulBajo = np.array([100,50,50],np.uint8)
        azulAlto = np.array([125,255,255],np.uint8)
        mask = cv2.inRange(hvs, azulBajo, azulAlto)
        imgcany = cv2.Canny(mask, 50, 70)
        contornos, _ = cv2.findContours(imgcany.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        final = cv2.drawContours(img, contornos, -1, (255,255,255))
        cv2.imshow('Contar', final)
        

        cv2.imshow('Original', img)
        cv2.imshow('HVS', hvs)
        cv2.imshow('Color detectado', mask)
        cv2.imshow('Detecciion', imgcany)
    
        if cv2.waitKey(1) & 0xFF == ord('s'):
            break

capture.release()
cv2.destroyAllWindows()