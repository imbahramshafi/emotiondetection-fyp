from ultralytics import YOLO
import cv2

# Load your trained YOLO model
model = YOLO("runs/detect/train3/weights/best.pt")

# Open webcam (0 = default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO on the frame
    results = model(frame)

    # Plot results onto the frame
    annotated_frame = results[0].plot()

    # Show the frame
    cv2.imshow("YOLO Webcam Face Detection", annotated_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
