import cv2
import numpy as np
import time as t

colors = {
    'yellow': [0, 255, 255], # yellow in BGR colorspace
    'blue': [255, 0, 0], # blue in BGR colorspace
    'red': [0, 0, 255], # red in BGR colorspace
    'green': [0, 255, 0] # green in BGR colorspace
}

def get_limits(color):
    c = np.uint8([[color]])  # BGR values
    hsvC = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)

    hue = hsvC[0][0][0]  # Get the hue value

    # Handle red hue wrap-around
    if hue >= 165:  # Upper limit for divided red hue
        lowerLimit = np.array([hue - 10, 100, 100], dtype=np.uint8)
        upperLimit = np.array([180, 255, 255], dtype=np.uint8)
    elif hue <= 15:  # Lower limit for divided red hue
        lowerLimit = np.array([0, 100, 100], dtype=np.uint8)
        upperLimit = np.array([hue + 10, 255, 255], dtype=np.uint8)
    else:
        lowerLimit = np.array([hue - 10, 100, 100], dtype=np.uint8)
        upperLimit = np.array([hue + 10, 255, 255], dtype=np.uint8)

    return lowerLimit, upperLimit

def process_contours(contours, color, frame):
        for c in contours:
            area = cv2.contourArea(c)
            if area > 500:
                M = cv2.moments(c)
                if M['m00'] != 0:
                    x = int(M['m10'] / M['m00'])
                    y = int(M['m01'] / M['m00'])
                    nuevoContorno = cv2.convexHull(c)
                    cv2.circle(frame, (x, y), 7, color, -1)
                    cv2.putText(frame, '{},{}'.format(x, y), (x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 1, cv2.LINE_AA)
                    cv2.drawContours(frame, [nuevoContorno], 0, color, 1)
                    return (x, y)
                else:
                    return (0, 0)


def get_color_name(X_coordinate, Y_coordinate, frame):
    height, width, _ = frame.shape
    if X_coordinate >= width or Y_coordinate >= height:
        return None
    
    center_pixel_bgr = frame[Y_coordinate, X_coordinate]
    blue_intensity, green_intensity, red_intensity = int(center_pixel_bgr[0]), int(center_pixel_bgr[1]), int(center_pixel_bgr[2])

    if red_intensity > 100 and green_intensity < 50 and blue_intensity < 50:
        color = 'red'
    elif red_intensity < 50 and green_intensity > 100 and blue_intensity < 50:
        color = 'green'
    elif red_intensity < 50 and green_intensity < 50 and blue_intensity > 100:
        color = 'blue'
    elif red_intensity > 100 and green_intensity > 100 and blue_intensity < 50:
        color = 'yellow'
    else:
        color = 'unknown'

    return color


cap = cv2.VideoCapture(0)
LINE_A = 410


while True:
    ret, frame = cap.read()

    hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    cv2.rectangle(frame, (LINE_A, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 1)

    for color_name, color_value in colors.items():
        lowerLimit, upperLimit = get_limits(color_value)
        color_mask = cv2.inRange(hsvImage, lowerLimit, upperLimit)
        contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            result = process_contours(contours, color_value, frame)
            if result is not None:
                x, y = result
                if x >= LINE_A:
                    reult_color = get_color_name(x, y, frame)
                    if reult_color != 'unknown':
                        print(reult_color)
                        print('Color: {}'.format(reult_color))
                        print('X: {}'.format(x))
                        print('Y: {}'.format(y))
                        # print data time
                        print(t.strftime('%H:%M:%S'))
                        print('------------------')

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()

cv2.destroyAllWindows()
