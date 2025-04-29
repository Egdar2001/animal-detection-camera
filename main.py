import cv2
import torch

# Load a pre-trained YOLOv8 model (Ultralytics library)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # 'yolov5s' is a small model, fast!

# You might want a model trained for animals specifically for better results!

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection
    results = model(frame)

    # Parse results
    for detection in results.xyxy[0]:  # each detection
        x1, y1, x2, y2, confidence, cls = detection
        label = model.names[int(cls)]

        # Draw rectangle
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
        cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    cv2.imshow('Animal Detector', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
