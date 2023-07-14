import cv2
import numpy as np

# Set the camera index and initialize the camera
CAMERA_INDEX = 2
cap = cv2.VideoCapture(CAMERA_INDEX)

# Set the frame width and height to 1280 and 720 pixels, respectively
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    # Read a frame from the camera and convert it from BGR to HSV color space
    _, frame = cap.read()
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Extract the height and width of the frame and calculate the center coordinates
    height, width, _ = frame.shape
    center_x = width // 2
    center_y = height // 2

    # Pick the pixel value at the center of the frame and extract the hue value
    center_pixel = hsv_frame[center_y, center_x]
    hue_value = center_pixel[0]

    # Determine the color of the object at the center of the frame based on the hue value
    if hue_value < 5:
        color = "Red"
    elif hue_value < 25:
        color = "Orange"
    elif hue_value < 33:
        color = "Yellow"
    elif hue_value < 78:
        color = "Green"
    elif hue_value < 125:
        color = "center_yan"
    elif hue_value < 165:
        color = "Blue"
    else:
        color = "Red"

    # Extract the BGR color values of the pixel located at the center of the frame
    center_pixel_bgr = frame[center_y, center_x]

    # Unpack the BGR intensities of the center pixel into separate variables
    blue_intensity, green_intensity, red_intensity = int(center_pixel_bgr[0]), int(center_pixel_bgr[1]), int(center_pixel_bgr[2])

    # Draw a text label indicating the color of the object at the center of the frame
    cv2.putText(frame, color, (10, 70),0, 1.5, (blue_intensity, green_intensity, red_intensity), 2, cv2.LINE_AA)


    # Draw a red circle at the center of the frame to highlight the location of the object
    cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), 3)

    # Display the processed frame in a window
    cv2.imshow('frame', frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera resources and close all windows
cap.release()
cv2.destroyAllWindows()