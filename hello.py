import cv2
from ultralytics import YOLO

# Load YOLOv8 model (you can use 'yolov8n.pt', 'yolov8s.pt', etc. for better speed/accuracy)
model = YOLO("yolov8n.pt")  # Downloaded automatically if not available

# Gun-related classes (YOLOv8 is trained on COCO dataset)
gun_classes = ['handgun', 'rifle', 'shotgun', 'weapon']  # Not all may be in COCO, so we'll rely on class IDs

# Define a function to check if detection is a gun
def is_gun(class_name):
    gun_keywords = ['gun', 'rifle', 'pistol', 'weapon', 'firearm']
    return any(keyword in class_name.lower() for keyword in gun_keywords)

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Run detection
    results = model(frame)[0]

    # Plot detections
    annotated_frame = results.plot()

    for box in results.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        class_name = model.names[class_id]

        if is_gun(class_name) and confidence > 0.6:
            print(f"🚨 Gun Detected: {class_name} ({confidence:.2f})")

    cv2.imshow("Gun Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
