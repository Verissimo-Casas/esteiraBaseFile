import cv2
import numpy as np
from flask import Flask, render_template, Response
import time
import requests

app = Flask(__name__)

# Variables para definir la ROI
roi_x = 300
roi_y = 200
roi_width = 400
roi_height = 300

# Variables para definir las líneas dentro de la ROI
LINE1_Y = 100
LINE2_Y = 200

def dibujar(mask, color, LINE1_Y, LINE2_Y, frame):
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

            # Check if the contour is within the ROI
            if roi_x <= x <= roi_x + roi_width and roi_y <= y <= roi_y + roi_height:
                cv2.circle(frame, (x, y), 7, color, -1)
                cv2.putText(frame, '{},{}'.format(x, y), (x + 10, y), font, 0.75, color, 1, cv2.LINE_AA)
                cv2.drawContours(frame, [nuevoContorno], 0, color, 1)

                # Check if object passes between the two lines
                if LINE1_Y <= y <= LINE2_Y:
                    count += 1

    return count


azulBajo = np.array([100,100,20],np.uint8)
azulAlto = np.array([125,255,255],np.uint8)

amarilloBajo = np.array([25,100,20],np.uint8)
amarilloAlto = np.array([45,255,255],np.uint8)

rojoBajo1 = np.array([0,100,20],np.uint8)
rojoAlto1 = np.array([5,255,255],np.uint8)

rojoBajo2 = np.array([170,100,20],np.uint8)
rojoAlto2 = np.array([179,255,255],np.uint8)

verdeBajo = np.array([35,100,20],np.uint8)
verdeAlto = np.array([85,255,255],np.uint8)

font = cv2.FONT_HERSHEY_SIMPLEX

def enviar_acumulado(endpoint, acumulado):
    data = {'acumulado': acumulado}  # Crear un diccionario con el acumulado
    headers = {'Content-Type': 'application/json'}  # Establecer el encabezado del contenido como JSON
    response = requests.post(endpoint, json=data, headers=headers)  # Enviar la solicitud POST al endpoint con los datos y encabezados

    if response.status_code == 200:
        print("Acumulado enviado con éxito")
    else:
        print("Error al enviar el acumulado")

@app.route('/')
def index():
    return render_template('index.html')


def generate_frames():
    cam = 0             # Camera
    width = 1080         # Largura
    height = 720        # Altura
    fps = 25            # FPS 25/30/50/60
    fourcc_type = 'mp4v'
    fourcc = cv2.VideoWriter_fourcc(*fourcc_type)

    codec = fourcc # MJPG

    brightness = 70     # Brilho
    contrast =  100     # Contraste
    saturation = 120    # Saturação

    # Câmera Hikvision não tem essas funçoes
    focus = 80           # Foco
    sharpness = 0       # Nitidez
    exposure = 100        # Exposiçãoq


    # FPS Teste
    start_time = time.time()
    display_time = 1
    fc = 0
    p_fps = 0

    # Conexão da câmera
    camera = cv2.VideoCapture(cam)

    # Configurar la camara
    camera.set(cv2.CAP_PROP_FOURCC, codec)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    camera.set(cv2.CAP_PROP_FPS, fps)
    camera.set(cv2.CAP_PROP_BRIGHTNESS, brightness)
    camera.set(cv2.CAP_PROP_CONTRAST, contrast)
    camera.set(cv2.CAP_PROP_SATURATION, saturation)
    camera.set(cv2.CAP_PROP_FOCUS, focus)
    camera.set(cv2.CAP_PROP_SHARPNESS, sharpness)
    camera.set(cv2.CAP_PROP_EXPOSURE, exposure)

    count_azul = 0
    count_amarillo = 0
    count_rojo = 0
    count_verde = 0
    totalCnts = 0

    while True:
        
        ret, frame = camera.read()
        if not ret:
            break

         # Dibujar ROI sobre la imagen completa
        cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_width, roi_y + roi_height), (0, 255, 0), 2)
        
        fc+=1

        TIME = time.time() - start_time
    
        if (TIME) >= display_time :
            p_fps = fc / (TIME)
            fc = 0
            start_time = time.time()
        
        fps_disp = "FPS: "+str(p_fps)[:5]
        
        frameHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        maskAzul = cv2.inRange(frameHSV, azulBajo, azulAlto)
        maskAmarillo = cv2.inRange(frameHSV, amarilloBajo, amarilloAlto)
        maskRojo1 = cv2.inRange(frameHSV, rojoBajo1, rojoAlto1)
        maskRojo2 = cv2.inRange(frameHSV, rojoBajo2, rojoAlto2)
        maskRojo = cv2.add(maskRojo1, maskRojo2)
        maskVerde = cv2.inRange(frameHSV, verdeBajo, verdeAlto)

        count_azul += dibujar(maskAzul, (255, 0, 0), LINE1_Y, LINE2_Y, frame)
        count_amarillo += dibujar(maskAmarillo, (0, 255, 255), LINE1_Y, LINE2_Y, frame)
        count_rojo += dibujar(maskRojo, (0, 0, 255), LINE1_Y, LINE2_Y, frame)
        count_verde += dibujar(maskVerde, (0, 255, 0), LINE1_Y, LINE2_Y, frame)
       
        totalCnts = count_azul + count_amarillo + count_rojo + count_verde

        

        # Dibujar líneas dentro de la ROI
        cv2.line(frame, (roi_x, roi_y + LINE1_Y), (roi_x + roi_width, roi_y + LINE2_Y), (0, 0, 255), 1)
        cv2.line(frame, (roi_x, roi_y + LINE1_Y), (roi_x + roi_width, roi_y + LINE2_Y), (0, 0, 255), 1)

        # Mostrar el contador dentro de la ROI
        #cv2.putText(frame, f"Contador: {count_total}", (roi_x + 10, roi_y + 30), font, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

        
        #frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        cv2.circle(frame, (30, 40), 15, (255, 0, 0), -1)
        cv2.circle(frame, (30, 80), 15, (0, 255, 255), -1)
        cv2.circle(frame, (30, 120), 15, (0, 0, 255), -1)
        cv2.circle(frame, (30, 160), 15, (0, 255, 0), -1)
        cv2.putText(frame, 'Total', (10,220), font, 1, (255, 255, 255), 2)
        frame = cv2.putText(frame, fps_disp, (800, 40), font, 0.7, (255, 255, 255), 2)

        cv2.putText(frame, str(count_azul), (65, 50), font, 0.75, (255, 255, 255), 2)
        cv2.putText(frame, str(count_amarillo), (65, 90), font, 0.75, (255, 255, 255), 2)
        cv2.putText(frame, str(count_rojo), (65, 130), font, 0.75, (255, 255, 255), 2)
        cv2.putText(frame, str(count_verde), (65, 170), font, 0.75, (255, 255, 255), 2)
        
        cv2.putText(frame, str(totalCnts), (105, 220), font, 0.75, (255, 255, 255), 2)
        
            
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

        # Enviar el acumulado cada 10 frames
        if count_azul % 10 == 0:
            enviar_acumulado('http://127.0.0.1:5000/endpoint', totalCnts)

        time.sleep(0.1)
        

    camera.release()

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
