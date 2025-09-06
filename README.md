# 🚦 Traffic Violation Detection

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-27338e?style=for-the-badge&logo=OpenCV&logoColor=white)
![YOLO](https://img.shields.io/badge/YOLO-00FFFF?style=for-the-badge&logo=YOLO&logoColor=black)

---

## 📝 Giới thiệu

**Dự án "Hệ Thống Nhận Diện Xe Vi Phạm Giao Thông"** được phát triển nhằm tự động hóa việc phát hiện xe vượt đèn đỏ và nhận diện biển số xe vi phạm tại các giao lộ. Ứng dụng giải quyết vấn đề ùn tắc và tai nạn giao thông bằng cách sử dụng công nghệ AI tiên tiến, hướng đến hỗ trợ quản lý giao thông hiệu quả và minh bạch.

---

## ✨ Tính năng chính

- Phát hiện trạng thái đèn giao thông (đỏ, vàng, xanh)
- Theo dõi và xác định xe vượt đèn đỏ dựa trên vùng cảnh báo (vạch đỏ, vạch xanh)
- Nhận diện biển số xe vi phạm bằng công nghệ YOLO
- Hiển thị kết quả thời gian thực trên giao diện Tkinter
- Lưu trữ thông tin vi phạm (biển số, thời gian, hình ảnh) vào file CSV

---

## ⚙️ Công nghệ sử dụng

- **YOLOv5 & YOLOv8**: Phát hiện và nhận diện đối tượng
- **OpenCV**: Xử lý ảnh và video
- **Tkinter**: Giao diện người dùng
- **Python**: Ngôn ngữ lập trình chính
- **NumPy, Pandas**: Xử lý dữ liệu

---

## 🚀 Cài đặt

### 1. Clone repository

```bash
git clone https://github.com/ThanhTam-dvl/Traffic_Violation_Detection.git
cd Traffic_Violation_Detection
```

### 2. Cài đặt dependencies

```bash
pip install -r requirement.txt
```

### 3. Download mô hình YOLO (nếu cần)

```bash
python download_yolo.py
```

---

## 🖥️ Cách sử dụng

### 1. Chạy ứng dụng chính

```bash
python detection.py
```

### 2. Chọn video/ảnh từ giao diện Tkinter

- Nhấn **"Chọn video"** để tải video giám sát
- Nhấn **"Test ảnh"** để kiểm tra nhận diện biển số

### 3. Vẽ vùng cảnh báo

Vẽ vùng cảnh báo (vạch đỏ, vạch xanh) bằng cách kéo thả chuột.

### 4. Xem kết quả

- Danh sách xe vi phạm hiển thị trên giao diện
- Video output được lưu ở `output_video.avi`

---

## 📁 Cấu trúc thư mục

```
TRAFFIC_VIOLATION_PROJECT/
├── __pycache__/
├── data/                          # Dữ liệu đầu vào (video, ảnh)
├── fx/                            # Hàm chức năng phụ
├── model/                         # Mô hình YOLO (yolov5m.pt, yolov8n.pt)
├── violations/                    # Dữ liệu vi phạm (violations.csv)
├── yolov5/                        # Thư mục YOLOv5
│   ├── detection.py               # Phát hiện đối tượng
│   └── lp_recognition.py          # Nhận diện biển số
├── txt/                           # File cấu hình hoặc log
├── detection.py                   # File chính chạy dự án
├── download_yolo.py               # Tải mô hình YOLO
├── lp_recognition.py              # Nhận diện biển số
├── main.py                        # File test nhận diện biển số
├── output_video.avi               # Video output
├── README.md                      # Tài liệu này
├── requirement.txt                # Dependencies
├── test_environment.py            # Kiểm tra môi trường
├── traffic_violation_detection.py # Logic chính phát hiện vi phạm
├── violations.csv                 # Danh sách vi phạm
├── yolov8m.pt                     # Mô hình YOLOv8 medium
└── yolov8n.pt                     # Mô hình YOLOv8 nano
```

---

## 📊 Kết quả và Demo

### Độ chính xác
- **Phát hiện đèn giao thông**: 92%
- **Xe vi phạm**: 90%
- **Nhận diện biển số**: 85%

### Hiệu suất
- Hệ thống chạy mượt mà trên video với scale 0.75
- Hiển thị danh sách vi phạm trên Tkinter
- Xem demo trong file `output_video.avi` hoặc hình ảnh từ tài liệu báo cáo

---

## ⚠️ Hạn chế và Hướng phát triển

### 🔴 Hạn chế
- Hiệu suất giảm trong điều kiện ánh sáng yếu
- Thời tiết xấu ảnh hưởng đến độ chính xác
- Biển số bị che khuất khó nhận diện
- Video/Camera mờ khó chụp được biển số

### 🔮 Hướng phát triển
- Tích hợp camera hồng ngoại và thuật toán tiền xử lý hình ảnh
- Lưu trữ dữ liệu vào cơ sở dữ liệu lâu dài
- Triển khai trên camera giao thông thực tế với GPU
- Cải thiện thuật toán nhận diện trong điều kiện khắc nghiệt

---

## 🤝 Đóng góp

1. Fork repository
2. Tạo branch mới:
   ```bash
   git checkout -b feature/ten-branch
   ```
3. Commit thay đổi:
   ```bash
   git commit -m 'Mô tả thay đổi'
   ```
4. Push lên branch:
   ```bash
   git push origin feature/ten-branch
   ```
5. Tạo Pull Request

---

## 📄 Giấy phép

Dự án được phát hành theo **MIT License**.

---

## 📧 Liên hệ

### 👥 Nhóm thực hiện:
- **Nguyễn Thành Tâm**  
  📧 Email: nguyenthanhtam10062004@gmail.com
- **Nguyễn Gia Chi Bảo**

---

## 🖼️ Demo Screenshots

*(Chưa cập nhật)*

- Giao diện chính
- Phát hiện vi phạm real-time  
- Kết quả nhận diện biển số
- Dashboard thống kê vi phạm
