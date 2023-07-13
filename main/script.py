import cv2

CAMERA = 0

cap = cv2.VideoCapture(CAMERA)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    _, frame = cap.read()
    hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    height, width, _ = frame.shape

    cx = int(width // 2)
    cy = int(height // 2)

    def Draw_Color_Box(CoordX: int, CoordY: int, Color: int, Color_Name: str):
        """
        Draw a color box with the color name
        CoordX: X coordinate of the box
        CoordY: Y coordinate of the box
        Color: HUE value of the color
        Color_Name: Name of the color
        """
        cv2.rectangle(frame, (CoordX, CoordY), (CoordX + 250, CoordY + 50), (Color, 0, 0), -1)
        cv2.putText(frame, Color_Name, (CoordX + 5, CoordY + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    Draw_Color_Box(10, 10, 0, "Red")

    # Pick pixel value at center of frame
    pixel_center = hsvImage[cy, cx]
    hue_value = pixel_center[0]

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

    cv2.circle(frame, (cx, cy), 5, (0, 0, 255), 3)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
