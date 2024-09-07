import cv2
import numpy as np

# Load YOLOv3 model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Load video
cap = cv2.VideoCapture("test_video.mp4")

# Define output video writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output_video.avi', fourcc, 30.0, (640, 480))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess frame
    frame = cv2.resize(frame, (640, 480))

    # Detect objects
    outputs = net.forward(frame)

    # Extract bounding boxes and class IDs
    boxes = []
    class_ids = []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id == 0:  # Filter by class ID (e.g., person)
                x, y, w, h = detection[0:4] * np.array([640, 480, 640, 480])
                boxes.append([x, y, w, h])
                class_ids.append(class_id)

    # Track objects
    tracker = cv2.MultiTracker_create()
    for box in boxes:
        tracker.add(cv2.TrackerKCF_create(), frame, box)

    # Draw bounding boxes and IDs
    for i, box in enumerate(boxes):
        x, y, w, h = box
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {i}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(frame, f"Label: {'Child' if class_ids[i] == 0 else 'Therapist'}", (x, y-25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Write output frame to video file
    out.write(frame)

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()