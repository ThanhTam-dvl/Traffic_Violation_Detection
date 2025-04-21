import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import time
import csv
import os
from datetime import datetime, timedelta
from collections import deque, defaultdict
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import threading
import queue

class TrafficViolationDetector:
    def __init__(self, model_path='yolov8m.pt'):
        self.model = YOLO(model_path)
        self.vehicle_classes = {1: 'xe dap', 2: 'o to', 3: 'xe may', 5: 'xe buyt', 7: 'xe tai'}
        self.traffic_light_states = {'red': (0, 0, 255), 'yellow': (0, 255, 255), 'green': (0, 255, 0), 'unknown': (128, 128, 128)}
        self.current_light_state = 'unknown'
        self.last_known_light_state = 'unknown'
        self.last_light_detection_time = datetime.now()
        
        # Default values for warning zone and lines
        self.warning_zone = {
            'x1': 300, 'y1': 400,  # Top-left corner
            'x2': 1600, 'y2': 800   # Bottom-right corner
        }
        
        # Green line: bottom line of warning zone
        self.green_line = {
            'x1': 300, 'y1': 750,  # Adjusted to match the visible green line in image
            'x2': 1600, 'y2': 760
        }
        
        # Red line: top line of warning zone (for complete violation)
        self.red_line = {
            'x1': 300, 'y1': 500,  # Adjusted to match the visible red line in image
            'x2': 1600, 'y2': 520
        }
        
        # For temporal smoothing of traffic light state
        self.state_history = deque(maxlen=5)
        self.state_history.extend(['unknown'] * 5)
        
        # Track vehicle violations
        self.vehicle_tracker = defaultdict(dict)
        self.violation_count = 0
        self.active_violations = set()  # Track currently violating vehicles
        self.processed_violations = set()  # Track violations we've already counted
        self.violation_timeout = 7  # Setting timeout to 7 seconds as requested
        
        # Vehicle buffer storage - to better track vehicles between frames
        self.vehicle_buffer = {}
        self.vehicle_persistence = 10  # Frames for vehicle tracking
        
        # Debug mode
        self.debug = False
    
    def detect_vehicles_and_lights(self, frame):
        scale_factor = 1.5
        frame_resized = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
        rgb_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        results = self.model(rgb_frame)
        
        vehicle_boxes = []
        traffic_lights = []
        annotated_frame = frame.copy()
        
        # Draw warning zone and lines
        self._draw_detection_zones(annotated_frame)
        
        # Track current frame vehicles to remove old violations
        current_frame_vehicles = set()
        
        # Process new detections and update vehicle tracking
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls)
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x1, y1, x2, y2 = int(x1/scale_factor), int(y1/scale_factor), int(x2/scale_factor), int(y2/scale_factor)
                
                if class_id in self.vehicle_classes and conf > 0.4:
                    # Calculate vehicle center point
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    
                    # Create unique vehicle ID based on position and class
                    vehicle_id = f"{self.vehicle_classes[class_id]}_{center_x//50}_{center_y//50}"
                    
                    # Try to match with existing vehicles in buffer
                    matched_id = self._match_vehicle(vehicle_id, center_x, center_y, class_id)
                    if matched_id:
                        vehicle_id = matched_id
                    
                    # Update or create vehicle buffer entry
                    self.vehicle_buffer[vehicle_id] = {
                        'position': (center_x, center_y),
                        'class_id': class_id,
                        'ttl': self.vehicle_persistence,
                        'bbox': (x1, y1, x2, y2)
                    }
                    
                    current_frame_vehicles.add(vehicle_id)
                    
                    vehicle_info = {
                        'class_id': class_id,
                        'class_name': self.vehicle_classes[class_id],
                        'bbox': (x1, y1, x2, y2),
                        'center': (center_x, center_y),
                        'confidence': conf,
                        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
                        'id': vehicle_id
                    }
                    vehicle_boxes.append(vehicle_info)
                    
                    # Use white border for cars, thinner line
                    color = (255, 255, 255) if class_id == 2 else self._get_class_color(class_id)
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 1)  # Reduced thickness
                    label = f"{self.vehicle_classes[class_id]} {conf:.2f}"
                    cv2.putText(annotated_frame, label, (x1, y1 - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)  # Smaller text
                    if self.debug:
                        cv2.circle(annotated_frame, (center_x, center_y), 3, (0, 255, 255), -1)
                        cv2.putText(annotated_frame, f"{vehicle_id.split('_')[0]}", (center_x, center_y), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                
                elif class_id == 9 and conf > 0.3:  # Traffic light
                    traffic_light_roi = frame[y1:y2, x1:x2]
                    light_state = self._detect_light_state(traffic_light_roi)
                    area = (x2 - x1) * (y2 - y1)
                    traffic_lights.append({
                        'bbox': (x1, y1, x2, y2),
                        'state': light_state,
                        'confidence': conf,
                        'area': area
                    })
        
        # Decrease TTL and remove expired vehicles
        self._update_vehicle_buffer()
        
        # Clean up vehicles no longer in frame
        self._clean_vehicle_tracker(current_frame_vehicles)
        
        # Check for violations
        for vehicle in vehicle_boxes:
            self._check_violations(vehicle, annotated_frame)
        
        # Process traffic lights
        if traffic_lights:
            if len(traffic_lights) > 1:
                traffic_lights.sort(key=lambda x: x['area'], reverse=True)
                closest_light = traffic_lights[0]
            else:
                closest_light = traffic_lights[0]
            
            new_state = closest_light['state']
            
            if new_state != 'unknown':
                self.last_known_light_state = new_state
                self.last_light_detection_time = datetime.now()
            else:
                time_since_last_detection = datetime.now() - self.last_light_detection_time
                if time_since_last_detection.total_seconds() < 5:
                    new_state = self.last_known_light_state
            
            self.state_history.append(new_state)
            smoothed_state = self._smooth_light_state()
            self.current_light_state = smoothed_state
            
            for tl in traffic_lights:
                x1, y1, x2, y2 = tl['bbox']
                state = tl['state']
                state_color = self.traffic_light_states[state]
                thickness = 2 if tl == closest_light else 1
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), state_color, thickness)
                cv2.putText(annotated_frame, f"Den tin hieu: {state}", 
                           (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, state_color, 1)
        else:
            time_since_last_detection = datetime.now() - self.last_light_detection_time
            if time_since_last_detection.total_seconds() < 5:
                new_state = self.last_known_light_state
            else:
                new_state = 'unknown'
            
            self.state_history.append(new_state)
            self.current_light_state = self._smooth_light_state()
        
        if self.debug:
            cv2.putText(annotated_frame, f"Light State: {self.current_light_state}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, self.traffic_light_states[self.current_light_state], 1)
            cv2.putText(annotated_frame, f"Active Violations: {len(self.active_violations)}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (0, 0, 255), 1)
            cv2.putText(annotated_frame, f"Total Vehicles: {len(vehicle_boxes)}", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (255, 255, 255), 1)
        
        self._display_status_info(annotated_frame)
        
        has_violations = len(self.active_violations) > 0
        
        return annotated_frame, vehicle_boxes, self.current_light_state, has_violations
    
    def _match_vehicle(self, vehicle_id, center_x, center_y, class_id):
        class_name = self.vehicle_classes[class_id]
        for existing_id, data in self.vehicle_buffer.items():
            if class_name in existing_id:
                ex_x, ex_y = data['position']
                distance = np.sqrt((center_x - ex_x)**2 + (center_y - ex_y)**2)
                if distance < 75:
                    return existing_id
        return None
    
    def _update_vehicle_buffer(self):
        keys_to_remove = []
        for vehicle_id, data in self.vehicle_buffer.items():
            data['ttl'] -= 1
            if data['ttl'] <= 0:
                keys_to_remove.append(vehicle_id)
        
        for key in keys_to_remove:
            del self.vehicle_buffer[key]
    
    def _draw_detection_zones(self, frame):
        zone_overlay = frame.copy()
        cv2.rectangle(zone_overlay, 
                     (self.warning_zone['x1'], self.warning_zone['y1']),
                     (self.warning_zone['x2'], self.warning_zone['y2']),
                     (0, 165, 255), -1)
        cv2.addWeighted(zone_overlay, 0.3, frame, 0.7, 0, frame)
        
        cv2.line(frame, 
                (self.green_line['x1'], self.green_line['y1']),
                (self.green_line['x2'], self.green_line['y2']),
                (0, 255, 0), 2)
        
        cv2.line(frame, 
                (self.red_line['x1'], self.red_line['y1']),
                (self.red_line['x2'], self.red_line['y2']),
                (0, 0, 255), 2)
        
        cv2.putText(frame, "Vach xanh", 
                   (self.green_line['x1'], self.green_line['y1'] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        cv2.putText(frame, "Vach do", 
                   (self.red_line['x1'], self.red_line['y1'] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    
    def _check_violations(self, vehicle, frame):
        vehicle_id = vehicle['id']
        center_x, center_y = vehicle['center']
        x1, y1, x2, y2 = vehicle['bbox']
        
        # Update vehicle data
        self.vehicle_tracker[vehicle_id]['last_bbox'] = (x1, y1, x2, y2)
        self.vehicle_tracker[vehicle_id]['last_seen'] = datetime.now()
        
        if 'green_crossed' not in self.vehicle_tracker[vehicle_id]:
            self.vehicle_tracker[vehicle_id] = {
                'green_crossed': False,
                'red_crossed': False,
                'violation_recorded': False,
                'warning_shown': False,
                'first_seen': datetime.now(),
                'position_history': [],
                'last_bbox': (x1, y1, x2, y2),
                'last_seen': datetime.now(),
                'violation_time': None
            }
        
        self.vehicle_tracker[vehicle_id]['position_history'].append((center_x, center_y))
        if len(self.vehicle_tracker[vehicle_id]['position_history']) > 10:
            self.vehicle_tracker[vehicle_id]['position_history'].pop(0)
        
        crossed_green = y2 >= self.green_line['y1']
        crossed_red = y1 <= self.red_line['y2']
        
        if crossed_green:
            self.vehicle_tracker[vehicle_id]['green_crossed'] = True
        if crossed_red:
            self.vehicle_tracker[vehicle_id]['red_crossed'] = True
        
        if self.current_light_state == 'red':
            if self.vehicle_tracker[vehicle_id]['green_crossed'] and self.vehicle_tracker[vehicle_id]['red_crossed']:
                self.active_violations.add(vehicle_id)
                
                if not self.vehicle_tracker[vehicle_id]['violation_recorded']:
                    self.vehicle_tracker[vehicle_id]['violation_recorded'] = True
                    self.vehicle_tracker[vehicle_id]['violation_time'] = datetime.now()
                    if vehicle_id not in self.processed_violations:
                        self.violation_count += 1
                        self.processed_violations.add(vehicle_id)
                
                violation_text = "XE VUOT DEN DO"
                cv2.putText(frame, violation_text, 
                           (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.6, (0, 0, 255), 1)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
            
            elif self.vehicle_tracker[vehicle_id]['green_crossed'] and not crossed_red:
                warning_text = "CANH BAO"
                cv2.putText(frame, warning_text, 
                           (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.6, (0, 255, 255), 1)
                self.vehicle_tracker[vehicle_id]['warning_shown'] = True
        
        if vehicle_id in self.active_violations:
            # Check if the violation should still be displayed (7 second timeout)
            if self.vehicle_tracker[vehicle_id].get('violation_time'):
                time_since_violation = datetime.now() - self.vehicle_tracker[vehicle_id]['violation_time']
                if time_since_violation.total_seconds() <= self.violation_timeout:
                    violation_text = "XE VUOT DEN DO"
                    cv2.putText(frame, violation_text, 
                            (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.6, (0, 0, 255), 1)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
                else:
                    # Remove from active violations after timeout
                    self.active_violations.remove(vehicle_id)
    
    def _clean_vehicle_tracker(self, current_frame_vehicles):
        current_time = datetime.now()
        vehicles_to_remove = set()
        
        for vehicle_id in self.active_violations:
            if vehicle_id not in current_frame_vehicles:
                if vehicle_id in self.vehicle_tracker:
                    last_seen = self.vehicle_tracker[vehicle_id].get('last_seen', current_time)
                    time_since_last_seen = (current_time - last_seen).total_seconds()
                    
                    # Check if violation has timed out (7 seconds)
                    violation_time = self.vehicle_tracker[vehicle_id].get('violation_time')
                    if violation_time:
                        time_since_violation = (current_time - violation_time).total_seconds()
                        if time_since_violation > self.violation_timeout:
                            vehicles_to_remove.add(vehicle_id)
                    
                    # Remove if not seen for more than 1 second
                    if time_since_last_seen > 1:
                        vehicles_to_remove.add(vehicle_id)
                else:
                    vehicles_to_remove.add(vehicle_id)
        
        self.active_violations -= vehicles_to_remove
        
        keys_to_remove = []
        for vehicle_id in self.vehicle_tracker:
            if vehicle_id not in current_frame_vehicles:
                last_seen = self.vehicle_tracker[vehicle_id].get('last_seen', current_time)
                if (current_time - last_seen).total_seconds() > 1:
                    keys_to_remove.append(vehicle_id)
        
        for key in keys_to_remove:
            del self.vehicle_tracker[key]
    
    def _display_status_info(self, frame):
        cv2.rectangle(frame, (10, 10), (350, 120), (0, 0, 0), -1)
        
        light_color = self.traffic_light_states[self.current_light_state]
        cv2.putText(frame, f"Den tin hieu:", (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"{self.current_light_state.upper()}", (20, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, light_color, 2)
        cv2.circle(frame, (250, 50), 12, light_color, -1)
        cv2.circle(frame, (250, 50), 14, (255, 255, 255), 1)
        
        cv2.putText(frame, f"Tong vi pham:", (20, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"{self.violation_count}", (250, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    def _smooth_light_state(self):
        state_counts = {'red': 0, 'yellow': 0, 'green': 0, 'unknown': 0}
        for state in self.state_history:
            state_counts[state] += 1
        
        if state_counts['unknown'] < 5:
            del state_counts['unknown']
        
        return max(state_counts, key=state_counts.get)
    
    def _detect_light_state(self, traffic_light_roi):
        if traffic_light_roi.size == 0:
            return 'unknown'
        
        hsv = cv2.cvtColor(traffic_light_roi, cv2.COLOR_BGR2HSV)
        
        red_lower1 = np.array([0, 100, 100])
        red_upper1 = np.array([15, 255, 255])
        red_lower2 = np.array([160, 100, 100])
        red_upper2 = np.array([180, 255, 255])
        
        green_lower = np.array([35, 50, 50])
        green_upper = np.array([90, 255, 255])
        
        yellow_lower = np.array([20, 100, 100])
        yellow_upper = np.array([35, 255, 255])
        
        red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
        red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        
        green_mask = cv2.inRange(hsv, green_lower, green_upper)
        yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
        
        total_pixels = traffic_light_roi.shape[0] * traffic_light_roi.shape[1]
        red_pixels = cv2.countNonZero(red_mask)
        green_pixels = cv2.countNonZero(green_mask)
        yellow_pixels = cv2.countNonZero(yellow_mask)
        
        red_percent = red_pixels / total_pixels * 100
        green_percent = green_pixels / total_pixels * 100
        yellow_percent = yellow_pixels / total_pixels * 100
        
        threshold = 2.0
        
        if max(red_percent, green_percent, yellow_percent) < threshold:
            return 'unknown'
        elif red_percent > green_percent and red_percent > yellow_percent:
            return 'red'
        elif green_percent > red_percent and green_percent > yellow_percent:
            return 'green'
        elif yellow_percent > red_percent and yellow_percent > green_percent:
            return 'yellow'
        else:
            return 'unknown'
    
    def reset_lines(self):
        # Reset to default line positions
        self.green_line = {
            'x1': 300, 'y1': 750,
            'x2': 1600, 'y2': 760
        }
        self.red_line = {
            'x1': 300, 'y1': 500,
            'x2': 1600, 'y2': 520
        }
        self.warning_zone = {
            'x1': 300, 'y1': 400,
            'x2': 1600, 'y2': 800
        }

    def _get_class_color(self, class_id):
        colors = {1: (0, 255, 0), 3: (255, 0, 0), 5: (255, 255, 0), 7: (0, 255, 255)}
        return colors.get(class_id, (128, 128, 128))


class TrafficViolationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Hệ thống phát hiện vi phạm giao thông")
        self.root.geometry("1280x800")
        
        # Initialize the detector
        self.detector = TrafficViolationDetector()
        
        # Initialize video variables
        self.video_path = None
        self.cap = None
        self.is_playing = False
        self.processing_thread = None
        self.frame_queue = queue.Queue(maxsize=30)
        self.stop_event = threading.Event()
        
        # Frame display variables
        self.current_frame = None
        self.display_frame = None
        self.frame_scale = 0.75
        
        # Variables for line drawing
        self.is_drawing = False
        self.drawing_line = None  # 'red' or 'green'
        self.drawing_start_point = None
        self.line_thickness = 2
        self.line_tmp = None
        
        # Create UI components
        self.create_ui()
        
        # Start the UI update loop
        self.update_ui()
    
    def create_ui(self):
        # Main frame
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Controls frame (top)
        self.controls_frame = ttk.Frame(self.main_frame)
        self.controls_frame.pack(fill=tk.X, side=tk.TOP, pady=5)
        
        # File selection button
        self.file_btn = ttk.Button(self.controls_frame, text="Tải video", command=self.load_video)
        self.file_btn.pack(side=tk.LEFT, padx=5)
        
        # Play/Pause button
        self.play_pause_btn = ttk.Button(self.controls_frame, text="Phát", command=self.toggle_play, state=tk.DISABLED)
        self.play_pause_btn.pack(side=tk.LEFT, padx=5)
        
        # Draw red line button
        self.draw_red_btn = ttk.Button(self.controls_frame, text="Vẽ vạch đỏ", command=lambda: self.start_drawing('red'))
        self.draw_red_btn.pack(side=tk.LEFT, padx=5)
        
        # Draw green line button
        self.draw_green_btn = ttk.Button(self.controls_frame, text="Vẽ vạch xanh", command=lambda: self.start_drawing('green'))
        self.draw_green_btn.pack(side=tk.LEFT, padx=5)
        
        # Reset lines button
        self.reset_lines_btn = ttk.Button(self.controls_frame, text="Đặt lại vạch", command=self.reset_lines)
        self.reset_lines_btn.pack(side=tk.LEFT, padx=5)
        
        # Status information
        self.status_label = ttk.Label(self.controls_frame, text="Chưa có video")
        self.status_label.pack(side=tk.RIGHT, padx=5)
        
        # Frame display area (center)
        self.canvas_frame = ttk.Frame(self.main_frame)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Canvas for displaying video
        self.canvas = tk.Canvas(self.canvas_frame, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Progress bar (bottom)
        self.progress_frame = ttk.Frame(self.main_frame)
        self.progress_frame.pack(fill=tk.X, side=tk.BOTTOM, pady=5)
        
        self.progress_bar = ttk.Progressbar(self.progress_frame, orient=tk.HORIZONTAL, length=100, mode='determinate')
        self.progress_bar.pack(fill=tk.X, padx=5)
        
        # Status bar (bottom)
        self.info_frame = ttk.Frame(self.main_frame)
        self.info_frame.pack(fill=tk.X, side=tk.BOTTOM, pady=5)
        
        self.violation_count_label = ttk.Label(self.info_frame, text="Vi pham: 0")
        self.violation_count_label.pack(side=tk.LEFT, padx=5)
        
        self.light_state_label = ttk.Label(self.info_frame, text="Den: chua xac dinh")
        self.light_state_label.pack(side=tk.LEFT, padx=5)
        
        # Bind canvas events for drawing
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)
    
    def load_video(self):
        file_path = filedialog.askopenfilename(
            title="Chọn video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov"), ("All files", "*.*")]
        )
        
        if file_path:
            self.video_path = file_path
            self.status_label.config(text=f"Đã tải: {os.path.basename(file_path)}")
            
            # Open video
            if self.cap is not None:
                self.cap.release()
            
            self.cap = cv2.VideoCapture(file_path)
            
            if not self.cap.isOpened():
                self.status_label.config(text="Lỗi: Không thể mở video")
                return
            
            # Get video properties
            self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.video_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.video_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Set canvas size based on video dimensions
            self.set_canvas_size()
            
            # Enable play button
            self.play_pause_btn.config(state=tk.NORMAL)
            
            # Read first frame to display
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame.copy()
                self.display_frame_on_canvas(frame)
    
    def set_canvas_size(self):
        # Calculate scaled dimensions
        scaled_width = int(self.video_width * self.frame_scale)
        scaled_height = int(self.video_height * self.frame_scale)
        
        # Update canvas size
        self.canvas.config(width=scaled_width, height=scaled_height)
    
    def toggle_play(self):
        if self.is_playing:
            self.is_playing = False
            self.play_pause_btn.config(text="Phát")
            self.stop_event.set()
        else:
            if self.cap is not None:
                self.is_playing = True
                self.play_pause_btn.config(text="Tạm dừng")
                self.stop_event.clear()
                self.start_processing()
    
    def start_processing(self):
        if self.processing_thread is not None and self.processing_thread.is_alive():
            self.processing_thread.join()
        
        self.processing_thread = threading.Thread(target=self.process_video)
        self.processing_thread.daemon = True
        self.processing_thread.start()
    
    def process_video(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_num = 0
        
        while self.is_playing and not self.stop_event.is_set():
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame_num += 1
            self.progress_bar['value'] = (frame_num / self.frame_count) * 100
            
            # Process frame
            processed_frame, _, light_state, _ = self.detector.detect_vehicles_and_lights(frame)
            
            # Update current frame
            self.current_frame = processed_frame.copy()
            
            # Put frame in queue for display
            if self.frame_queue.full():
                self.frame_queue.get()
            self.frame_queue.put(processed_frame)
            
            # Control playback speed
            time.sleep(1/self.fps)
        
        # When video ends or is stopped
        if not self.stop_event.is_set():
            self.is_playing = False
            self.play_pause_btn.config(text="Phát")
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    def display_frame_on_canvas(self, frame):
        if frame is None:
            return
        
        # Convert frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize frame to fit canvas
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            return
        
        frame_resized = cv2.resize(frame_rgb, (canvas_width, canvas_height))
        
        # Convert to ImageTk format
        img = Image.fromarray(frame_resized)
        imgtk = ImageTk.PhotoImage(image=img)
        
        # Update canvas
        self.canvas.imgtk = imgtk
        self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
    
    def update_ui(self):
        # Update frame display
        if not self.frame_queue.empty():
            frame = self.frame_queue.get()
            self.display_frame_on_canvas(frame)
        
        # Update status information
        self.violation_count_label.config(text=f"Vi phạm: {self.detector.violation_count}")
        self.light_state_label.config(text=f"Đèn: {self.detector.current_light_state.upper()}")
        
        # Schedule next update
        self.root.after(50, self.update_ui)
    
    def start_drawing(self, line_type):
        self.is_drawing = True
        self.drawing_line = line_type
        self.status_label.config(text=f"Đang vẽ vạch {'đỏ' if line_type == 'red' else 'xanh'}. Nhấp và kéo trên video")
    
    def on_canvas_click(self, event):
        if self.is_drawing:
            self.drawing_start_point = (event.x, event.y)
    
    def on_canvas_drag(self, event):
        if self.is_drawing and self.drawing_start_point:
            # Create a temporary line while dragging
            if self.line_tmp:
                self.canvas.delete(self.line_tmp)
            
            x1, y1 = self.drawing_start_point
            color = 'red' if self.drawing_line == 'red' else 'green'
            self.line_tmp = self.canvas.create_line(
                x1, y1, event.x, event.y,
                fill=color, width=self.line_thickness
            )
    
    def on_canvas_release(self, event):
        if self.is_drawing and self.drawing_start_point:
            x1, y1 = self.drawing_start_point
            x2, y2 = event.x, event.y
            
            # Get canvas dimensions and video dimensions
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            video_width = self.video_width
            video_height = self.video_height
            
            # Calculate scaling factors
            scale_x = video_width / canvas_width
            scale_y = video_height / canvas_height
            
            # Convert canvas coordinates to video coordinates
            vid_x1 = int(x1 * scale_x)
            vid_y1 = int(y1 * scale_y)
            vid_x2 = int(x2 * scale_x)
            vid_y2 = int(y2 * scale_y)
            
            # Update the detector's line
            if self.drawing_line == 'red':
                self.detector.red_line = {
                    'x1': vid_x1, 'y1': vid_y1,
                    'x2': vid_x2, 'y2': vid_y2
                }
            else:
                self.detector.green_line = {
                    'x1': vid_x1, 'y1': vid_y1,
                    'x2': vid_x2, 'y2': vid_y2
                }
            
            # Update warning zone between the lines
            self.detector.warning_zone = {
                'x1': min(vid_x1, vid_x2),
                'y1': min(vid_y1, vid_y2),
                'x2': max(vid_x1, vid_x2),
                'y2': max(vid_y1, vid_y2)
            }
            
            # Clean up
            if self.line_tmp:
                self.canvas.delete(self.line_tmp)
                self.line_tmp = None
            
            self.is_drawing = False
            self.drawing_start_point = None
            self.status_label.config(text=f"Đã vẽ vạch {'đỏ' if self.drawing_line == 'red' else 'xanh'}")
    
    def reset_lines(self):
        self.detector.reset_lines()
        self.status_label.config(text="Đã đặt lại vạch về mặc định")
    
    def on_closing(self):
        # Stop video processing
        self.is_playing = False
        self.stop_event.set()
        
        # Wait for processing thread to finish
        if self.processing_thread is not None and self.processing_thread.is_alive():
            self.processing_thread.join()
        
        # Release video capture
        if self.cap is not None:
            self.cap.release()
        
        # Destroy the window
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = TrafficViolationApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
       