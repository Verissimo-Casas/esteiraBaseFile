import cv2

CAMERA = 0
cap = cv2.VideoCapture(CAMERA)

def draw_contours(frame, contours, color):
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1000:
            cv2.drawContours(frame, [cnt], -1, color, 3)

while True:
    frame = cap.read()[1]
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

cap.release()
cv2.destroyAllWindows()