import cv2
import numpy as np
import winsound

# Load pre-trained model and config file
model = "MobileNetSSD_deploy.caffemodel"
config = "MobileNetSSD_deploy.prototxt.txt"
net = cv2.dnn.readNetFromCaffe(config, model)

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Parameters for processing
frame_skip = 1  # Process every frame (no skipping)
resize_factor = 0.75  # Resize the frame slightly to speed up processing

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame from webcam.")
        break

    frame_count += 1

    # Resize the frame to speed up processing
    small_frame = cv2.resize(frame, None, fx=resize_factor, fy=resize_factor)

    # Prepare the frame for human detection
    h, w = small_frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(small_frame, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    # Loop over the detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:
            idx = int(detections[0, 0, i, 1])

            # Only consider human class (class id 15 for MobileNetSSD)
            if idx == 15:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # Adjust bounding box to match the original frame size
                startX, startY = int(startX / resize_factor), int(startY / resize_factor)
                endX, endY = int(endX / resize_factor), int(endY / resize_factor)

                # Draw the bounding box with a red outline around the detected human
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
                print(f"Human detected at [{startX}, {startY}, {endX}, {endY}] on frame {frame_count}")

                # Beep sound when motion is detected
                winsound.Beep(1000, 500)

    # Display the resulting frame
    cv2.imshow('Frame', frame)

    # Break loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Manual exit triggered.")
        break

# Release the capture and destroy all windows
cap.release()
cv2.destroyAllWindows()
print("Real-time video processing completed.")
