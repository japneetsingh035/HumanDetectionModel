import cv2
import numpy as np  # Import numpy and alias it as np
import winsound
# Load pre-trained model and config file
model = "MobileNetSSD_deploy.caffemodel"
config = "MobileNetSSD_deploy.prototxt.txt"

# Initialize the network
net = cv2.dnn.readNetFromCaffe(config, model)

# Replace 'your_video_file.mp4' with the path to your recorded video
video_path = 'QA-Test-Case-01.mp4'
cap = cv2.VideoCapture(video_path)

# Get the frames per second (FPS) of the video
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Video FPS: {fps}")

# Parameters for processing
frame_skip = int(fps // 2)  # Skip half the frames, adjust as needed
resize_factor = 0.5  # Resize the frame by 50%

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video or failed to read frame.")
        break

    frame_count += 1

    # Skip frames to speed up processing
    if frame_count % frame_skip != 0:
        continue

    # Resize the frame to speed up processing
    frame = cv2.resize(frame, None, fx=resize_factor, fy=resize_factor)

    # Prepare the frame for human detection
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
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

                # Draw the bounding box with a red outline around the detected human
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
                print(f"Human detected at [{startX}, {startY}, {endX}, {endY}] on frame {frame_count}")

                # Beep sound when motion is detected
                winsound.Beep(1000, 500)

    # Display the resulting frame (Optional)
    cv2.imshow('Frame', frame)

    # Break loop if 'q' is pressed
    if cv2.waitKey(30) & 0xFF == ord('q'):
        print("Manual exit triggered.")
        break

# Release the capture and destroy all windows
cap.release()
cv2.destroyAllWindows()
print("Video processing completed.")