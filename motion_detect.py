import cv2
import winsound

# Initialize video capture (0 is the default camera)
cap = cv2.VideoCapture(0)

# Initialize the background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Apply background subtraction
    fgmask = fgbg.apply(frame)

    # Find contours (motion detection)
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # If contours are found, there's motion in the video
    if len(contours) > 0:
        # Beep sound when motion is detected
        winsound.Beep(1000, 500)  # Frequency 1000Hz, Duration 500ms
        break  # Optional: exit the loop after first detection

    # Display the resulting frame
    cv2.imshow('Frame', frame)

    # Break loop if 'q' is pressed
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Release the capture and destroy all windows
cap.release()
cv2.destroyAllWindows()
