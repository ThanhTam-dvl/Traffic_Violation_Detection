from ultralytics import YOLO
model = YOLO('yolov8n.pt')  # Automatically downloads yolov8n.pt
print("YOLOv8 model downloaded successfully!")