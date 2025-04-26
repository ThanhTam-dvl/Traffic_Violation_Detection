import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import threading
import cv2
from lp_recognition import process_image, process_video_frame, process_webcam


class VideoProcessor:
    def __init__(self):
        self.video_capture = None
        self.is_playing = False
        self.current_frame = None
        self.plates_in_video = []

    def open_video(self, filepath):
        self.video_capture = cv2.VideoCapture(filepath)
        self.is_playing = True
        self.play_video()

    def play_video(self):
        if self.video_capture is None or not self.is_playing:
            return

        ret, frame = self.video_capture.read()
        if ret:
            # Xử lý frame để nhận dạng biển số
            processed_frame, plates = process_video_frame(frame)
            
            if plates:
                self.plates_in_video.extend(plates)
                plate_result_label.config(text="\n".join([f"Biển số: {plate}" for plate in self.plates_in_video]))
            
            # Hiển thị frame đã xử lý
            self.current_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(self.current_frame)
            img = img.resize((600, 400))  # Kích thước lớn hơn để xem rõ hơn
            img_tk = ImageTk.PhotoImage(image=img)
            panel.config(image=img_tk)
            panel.image = img_tk
            
            # Lặp lại sau 25ms (40 FPS)
            panel.after(25, self.play_video)
        else:
            self.stop_video()

    def stop_video(self):
        self.is_playing = False
        if self.video_capture is not None:
            self.video_capture.release()
        video_label.config(text="Video đã kết thúc")


video_processor = VideoProcessor()

def select_image():
    filepath = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.png *.jpeg")])
    if filepath:
        img_label.config(text=f"Đã chọn: {filepath}")
        display_image(filepath)
        plates = process_image(filepath)
        display_plate_results(plates)


def display_image(filepath):
    img = Image.open(filepath)
    img = img.resize((600, 400))  # Kích thước lớn hơn
    img_tk = ImageTk.PhotoImage(img)
    panel.config(image=img_tk)
    panel.image = img_tk


def display_plate_results(plates):
    if plates:
        result_text = "\n".join([f"Biển số: {plate}" for plate in plates])
    else:
        result_text = "Không nhận dạng được biển số!"
    plate_result_label.config(text=result_text)


def start_webcam():
    threading.Thread(target=process_webcam, daemon=True).start()


def select_video():
    filepath = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi *.mov")])
    if filepath:
        video_label.config(text=f"Đang phát: {filepath.split('/')[-1]}")
        video_processor.plates_in_video = []  # Reset kết quả biển số
        plate_result_label.config(text="Đang xử lý video...")
        threading.Thread(target=lambda: video_processor.open_video(filepath), daemon=True).start()


# Tạo giao diện Tkinter
app = tk.Tk()
app.title("Hệ thống nhận dạng biển số")
app.geometry("800x800")  # Cửa sổ lớn hơn

# Tiêu đề
title = tk.Label(app, text="Hệ Thống Nhận Dạng Biển Số Xe", font=("Arial", 18, "bold"))
title.pack(pady=10)

# Nút chọn ảnh, video và bật webcam
btn_frame = tk.Frame(app)
btn_frame.pack(pady=10)

btn_select = tk.Button(
    btn_frame, text="Chọn Hình Ảnh", command=select_image, width=20, bg="lightblue"
)
btn_select.pack(side="left", padx=5)

btn_video = tk.Button(
    btn_frame, text="Chọn Video", command=select_video, width=20, bg="orange"
)
btn_video.pack(side="left", padx=5)

btn_webcam = tk.Button(
    btn_frame, text="Bật Webcam", command=start_webcam, width=20, bg="lightgreen"
)
btn_webcam.pack(side="left", padx=5)

# Hiển thị thông tin hình ảnh hoặc video
img_label = tk.Label(app, text="Chưa chọn hình ảnh")
img_label.pack()

video_label = tk.Label(app, text="Chưa chọn video")
video_label.pack()

# Panel hiển thị ảnh/video
panel = tk.Label(app)
panel.pack()

# Kết quả biển số
plate_result_label = tk.Label(
    app, text="", font=("Arial", 12), fg="blue", justify="left", wraplength=600
)
plate_result_label.pack(pady=10)

# Nút thoát
btn_exit = tk.Button(app, text="Thoát", command=app.quit, width=20, bg="red", fg="white")
btn_exit.pack(pady=20)

# Chạy ứng dụng
app.mainloop()