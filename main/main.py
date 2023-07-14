import cv2

CAMERA = 2
cap = cv2.VideoCapture(CAMERA)

def find_contours(hsv_frame):

while True:
    frame = cap.read()[1]
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

cap.release()
cv2.destroyAllWindows()