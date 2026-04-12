from ultralytics import YOLO
import cv2
import os

# Load your trained model
model = YOLO("runs/detect/train3/weights/best.pt")

# Run prediction (your image path here)
results = model.predict(source="C:/Users/bahra/Downloads/test.jpg", save=True)

# YOLO saves the output to: runs/detect/predictX/
save_dir = results[0].save_dir
out_path = os.path.join(save_dir, os.path.basename("C:/Users/bahra/Downloads/test.jpg"))

# Load the saved image and display it
img = cv2.imread(out_path)
cv2.imshow("YOLO Detection", img)

cv2.waitKey(0)  # <-- keeps window open until you press a key
cv2.destroyAllWindows()
