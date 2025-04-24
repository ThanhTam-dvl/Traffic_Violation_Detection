import cv2
import torch
from ultralytics import YOLO
from function.helper import read_plate
from function.utils_rotate import deskew

# Tải mô hình YOLO
yolo_LP_detect = torch.hub.load(
    'yolov5', 'custom', path='model/LP_detector.pt', force_reload=True, source='local'
)
yolo_license_plate = torch.hub.load(
    'yolov5', 'custom', path='model/LP_ocr.pt', force_reload=True, source='local'
)
yolo_license_plate.conf = 0.6


def process_image(image_path):
    img = cv2.imread(image_path)
    plates = yolo_LP_detect(img, size=640)
    list_plates = plates.pandas().xyxy[0].values.tolist()
    detected_plates = []  # Danh sách biển số đã nhận dạng

    if not list_plates:
        print("Không phát hiện biển số!")
    else:
        for plate in list_plates:
            x, y, w, h = int(plate[0]), int(plate[1]), int(plate[2] - plate[0]), int(plate[3] - plate[1])
            crop_img = img[y:y + h, x:x + w]
            lp_text = read_plate(yolo_license_plate, deskew(crop_img, 1, 0))
            if lp_text != "unknown":
                detected_plates.append(lp_text)
                # Ghi biển số lên ảnh
                cv2.putText(img, lp_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                # Vẽ khung bao
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Lưu ảnh kết quả
    cv2.imwrite("images/output.jpg", img)
    print("Đã xử lý và lưu tại images/output.jpg")

    return detected_plates  # Trả về danh sách biển số


def process_webcam():
    cap = cv2.VideoCapture(1)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        plates = yolo_LP_detect(frame, size=640)
        list_plates = plates.pandas().xyxy[0].values.tolist()

        for plate in list_plates:
            x, y, w, h = int(plate[0]), int(plate[1]), int(plate[2] - plate[0]), int(plate[3] - plate[1])
            crop_img = frame[y:y + h, x:x + w]
            lp_text = read_plate(yolo_license_plate, deskew(crop_img, 1, 0))
            if lp_text != "unknown":
                cv2.putText(frame, lp_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.imshow("Webcam - Nhận dạng biển số", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    output_path = "videos/output.avi"
    
    # Lấy thông số video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Định nghĩa codec và tạo đối tượng VideoWriter để lưu video đầu ra
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Nhận diện biển số trên từng khung hình
        plates = yolo_LP_detect(frame, size=600)
        list_plates = plates.pandas().xyxy[0].values.tolist()

        for plate in list_plates:
            x, y, w, h = int(plate[0]), int(plate[1]), int(plate[2] - plate[0]), int(plate[3] - plate[1])
            crop_img = frame[y:y + h, x:x + w]
            lp_text = read_plate(yolo_license_plate, deskew(crop_img, 1, 0))
            if lp_text != "unknown":
                # Hiển thị biển số trên khung hình
                cv2.putText(frame, lp_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                
        # Lưu khung hình đã xử lý vào video đầu ra
        out.write(frame)
        
        # Hiển thị video trong thời gian thực
        cv2.imshow("Video - Nhận dạng biển số", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Video đã xử lý và lưu tại {output_path}")
