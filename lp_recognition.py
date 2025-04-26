import cv2
import torch
from ultralytics import YOLO
from function.helper import read_plate
from function.utils_rotate import deskew

# Tải mô hình YOLO (chỉ tải 1 lần khi import)
yolo_LP_detect = None
yolo_license_plate = None

def load_models():
    global yolo_LP_detect, yolo_license_plate
    if yolo_LP_detect is None:
        yolo_LP_detect = torch.hub.load(
            'yolov5', 'custom', path='model/LP_detector.pt', force_reload=True, source='local'
        )
    if yolo_license_plate is None:
        yolo_license_plate = torch.hub.load(
            'yolov5', 'custom', path='model/LP_ocr.pt', force_reload=True, source='local'
        )
        yolo_license_plate.conf = 0.6

# Load models khi import module
load_models()

def process_image(image_path):
    """Xử lý ảnh tĩnh và trả về biển số nhận dạng được"""
    img = cv2.imread(image_path)
    if img is None:
        print(f"Không thể đọc ảnh từ {image_path}")
        return []
    
    plates = yolo_LP_detect(img, size=640)
    list_plates = plates.pandas().xyxy[0].values.tolist()
    detected_plates = []

    for plate in list_plates:
        x, y, w, h = int(plate[0]), int(plate[1]), int(plate[2] - plate[0]), int(plate[3] - plate[1])
        crop_img = img[y:y + h, x:x + w]
        lp_text = read_plate(yolo_license_plate, deskew(crop_img, 1, 0))
        if lp_text != "unknown":
            detected_plates.append(lp_text)
            cv2.putText(img, lp_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imwrite("images/output.jpg", img)
    return detected_plates

def process_video_frame(frame):
    """Xử lý từng frame video và trả về frame đã xử lý cùng danh sách biển số"""
    plates = yolo_LP_detect(frame, size=640)
    list_plates = plates.pandas().xyxy[0].values.tolist()
    detected_plates = []

    for plate in list_plates:
        x, y, w, h = int(plate[0]), int(plate[1]), int(plate[2] - plate[0]), int(plate[3] - plate[1])
        crop_img = frame[y:y + h, x:x + w]
        lp_text = read_plate(yolo_license_plate, deskew(crop_img, 1, 0))
        if lp_text != "unknown":
            detected_plates.append(lp_text)
            cv2.putText(frame, lp_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    return frame, detected_plates

def process_webcam():
    """Xử lý video từ webcam"""
    cap = cv2.VideoCapture(0)  # Thường camera mặc định là 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame, plates = process_video_frame(frame)
            
            cv2.imshow("Webcam - Nhận dạng biển số", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

def process_video(video_path, output_path="videos/output.avi"):
    """Xử lý video file và lưu kết quả"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Không thể mở video: {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            processed_frame, _ = process_video_frame(frame)
            out.write(processed_frame)
            
            cv2.imshow("Video - Nhận dạng biển số", processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"Video đã xử lý được lưu tại: {output_path}")