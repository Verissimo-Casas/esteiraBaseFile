import cv2

CAMERA = 0 # 0 = Camara integrada, 1 = Camara externa



cap = cv2.VideoCapture(CAMERA)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 620)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FOCUS, 100)


while True:
    _, frame = cap.read()
    hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    height, width, _ = frame.shape

    cx = int(width // 2)
    cy = int(height // 2)

    # Pick pixel value at center of frame
    pixel_center = hsvImage[cy, cx]
    hue_value = pixel_center[0]

    # draw(x, y, color int = 5 name string = "Red")
    # draw(x, y, color int = 25 name string = "Orange")
    # draw(x, y, color int = 33 name string = "Yellow")

    color = "Undefined"

    if hue_value < 5:
        color = "Red"
    elif hue_value < 25:
        color = "Orange"
    elif hue_value < 33:
        color = "Yellow"
    elif hue_value < 78:
        color = "Green"
    elif hue_value < 125:
        color = "Cyan"
    elif hue_value < 165:
        color = "Blue"
    else:
        color = "Red"

    pixel_center_bgr = frame[cy, cx]
    b, g, r = int(pixel_center_bgr[0]), int(pixel_center_bgr[1]), int(pixel_center_bgr[2])

    # cv2.putText(frame, f"{color} ({b}, {g}, {r})", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(frame, color, (10, 70), 0, 1.5, (b, g, r), 2, cv2.LINE_AA)

    rect_top_left = (cx - 180, cy - 500)
    rect_bottom_right = (cx + 5, cy + 5)
    cv2.rectangle(frame, rect_top_left, rect_bottom_right, (0, 0, 255), 3)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
