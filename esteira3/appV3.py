import cv2
import numpy as np
from flask import Flask, render_template, Response
import time
import requests

app = Flask(__name__)


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
    cv2.putText(frame, '{},{}'.format(x, y), (x + 10, y), font, 0.75, color, 1, cv2.LINE_AA)

def draw_contour(frame, contour, color):
    nuevo_contorno = cv2.convexHull(contour)
    cv2.drawContours(frame, [nuevo_contorno], 0, color, 1)

def count_objects_between_lines(contours, line1, line2):
    count = 0
    for c in contours:
        x, y = calculate_centroid(c)
        if line1 > y > line2:
            count += 1
    return count

def dibujar(mask, color, line1, line2, frame):
    contours = find_contours(mask)
    for c in contours:
        area = cv2.contourArea(c)
        if area > 1000:
            x, y = calculate_centroid(c)
            draw_circle_and_text(frame, x, y, color)
            draw_contour(frame, c, color)

    count = count_objects_between_lines(contours, line1, line2)
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
    cam = 0            # Camera
    width = 1080         # Ancho
    height = 680        # Alto
    fps = 60            # FPS 25/30/50/60
    fourcc_type = 'mp4v'
    fourcc = cv2.VideoWriter_fourcc(*fourcc_type)

    codec = fourcc # MJPG

    brightness = 25     # Brillo
    contrast =  15     # Contraste
    saturation = 20    # Saturacion

    focus = 100           # Foco
    sharpness = 3       # Nitidez
    exposure = 0        # Expocision


    # FPS Teste
    start_time = time.time()
    display_time = 1
    fc = 0
    p_fps = 0

    

    # Configurar la camara
    camera = cv2.VideoCapture(cam)
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


    LINE1_Y = 110
    LINE2_Y = 100

    count_azul = 0
    count_amarillo = 0
    count_rojo = 0
    count_verde = 0
    totalCnts = 0

    while True:
        
        ret, frame = camera.read()
        if not ret:
            break
        
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

        count_azul += dibujar(maskAzul, (255, 0, 0), LINE1_Y, LINE2_Y, frame)
        count_amarillo += dibujar(maskAmarillo, (0, 255, 255), LINE1_Y, LINE2_Y, frame)


        totalCnts = count_azul + count_amarillo + count_rojo + count_verde


        cv2.line(frame, (0, LINE1_Y), (frame.shape[1], LINE1_Y), (0, 0, 255), 1)
        cv2.line(frame, (0, LINE2_Y), (frame.shape[1], LINE2_Y), (0, 0, 255), 1)

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