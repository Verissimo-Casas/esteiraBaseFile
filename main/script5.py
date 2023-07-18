import cv2
import numpy as np

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
        if area > 1500:
            M = cv2.moments(c)
            if M["m00"] == 0:
                M["m00"] = 1
            x = int(M["m10"] / M["m00"])
            y = int(M['m01'] / M['m00'])
            nuevoContorno = cv2.convexHull(c)
            cv2.circle(frame, (x, y), 7, color, -1)
            cv2.putText(frame, '{},{}'.format(x, y), (x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 1, cv2.LINE_AA)
            cv2.drawContours(frame, [nuevoContorno], 0, color, 1)

            return x, y

def get_color_name(X_coordinate: int, Y_coordinate: int, frame):
    '''Returns the color name of the pixel located at the specified coordinates'''
    center_pixel_bgr = frame[X_coordinate, Y_coordinate]
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

def draw_line(x, y, frame):
    cv2.line(frame, (x, y), (210, 210), (0, 0, 0), 2)

    return x, y

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()

    hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    for color_name, color_value in colors.items():
        lowerLimit, upperLimit = get_limits(color_value)
        color_mask = cv2.inRange(hsvImage, lowerLimit, upperLimit)
        contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        process_contours(contours, color_value, frame)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()

cv2.destroyAllWindows()
