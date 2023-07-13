import cv2
import numpy as np
from flask import Flask, render_template, Response
import time

app = Flask(__name__)

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
            cv2.circle(mask, (x, y), 7, color, -1)
            cv2.putText(mask, '{},{}'.format(x, y), (x + 10, y), font, 0.75, color, 1, cv2.LINE_AA)
            cv2.drawContours(mask, [nuevoContorno], 0, color, 1)

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
LINE2_Y = 205

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

def capture(sources):
    ''' esta función captura video de una fuente específica y devuelve los frames como imágenes JPEG 
    codificadas a través de un generador. Utiliza un bucle infinito y un bloque try-except para 
    manejar errores y garantizar que la captura de video continúe a pesar de cualquier problema. '''
    while True:
        try:
            cap = cv2.VideoCapture(sources)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
            #salida = cv2.VideoWriter('videoSalida.avi',cv2.VideoWriter_fourcc(*'XVID'),20.0,(640,480))
            while True:
                ret, frame = cap.read()
                if frame is None:
                    break

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
                count_naranja += dibujar(maskNaranja, (0, 165, 255), LINE1_Y, LINE2_Y)
                count_marron += dibujar(maskMarron, (42, 42, 165), LINE1_Y, LINE2_Y)
                count_gris += dibujar(maskGris, (128, 128, 128), LINE1_Y, LINE2_Y)
                count_violeta += dibujar(maskVioleta, (238, 130, 238), LINE1_Y, LINE2_Y)
                count_blanco += dibujar(maskBlanco, (255, 255, 255), LINE1_Y, LINE2_Y)

                totalCnts = count_azul + count_amarillo + count_rojo + count_verde + count_naranja + count_marron + count_gris + count_violeta + count_blanco

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
                cv2.putText(frame, str(count_naranja), (65, 210), font, 0.75, (0, 0, 0), 1)
                cv2.putText(frame, str(count_marron), (65, 250), font, 0.75, (0, 0, 0), 1)
                cv2.putText(frame, str(count_gris), (65, 290), font, 0.75, (0, 0, 0), 1)
                cv2.putText(frame, str(count_violeta), (65, 330), font, 0.75, (0, 0, 0), 1)
                cv2.putText(frame, str(count_blanco), (65, 370), font, 0.75, (0, 0, 0), 1)

                cv2.putText(frame, str(totalCnts), (55, 410), font, 0.75, (0, 0, 0), 1)

            
                ret, jpeg = cv2.imencode('.jpg', frame)
                if ret:
                            
                            yield (b'--frame\r\n'
                                b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
                            cap.release()
        except:
            time.sleep(5)
            return jpeg.tobytes()

@app.route('/video_feed')
def video_feed():
    return Response(capture(source),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
    
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    #source = 'admin:asdqwe123@10.0.0.10:554/camrealmonitor?cannel1&subtype=0'
    #source = 'rtsp://administrator:asdqwe123@172.16.18.89:554/defaultPrimary?mtu=1440&streamType=u'
    source = 0
    app.run(host='0.0.0.0', debug=True)
