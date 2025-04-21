import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import time
import csv
from datetime import datetime, timedelta
from collections import deque, defaultdict

class TrafficViolationDetector:
    def __init__(self, model_path='yolov8m.pt'):
        self.model = YOLO(model_path)
        self.vehicle_classes = {1: 'bicycle', 2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
        self.traffic_light_states = {'red': (0, 0, 255), 'yellow': (0, 255, 255), 'green': (0, 255, 0), 'unknown': (128, 128, 128)}
        self.current_light_state = 'unknown'
        self.last_known_light_state = 'unknown'
        self.last_light_detection_time = datetime.now()
        
        # Define warning zone and lines
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
        self.violation_timeout = 3  # Reduced timeout to remove warnings faster
        
        # Vehicle buffer storage - to better track vehicles between frames
        self.vehicle_buffer = {}
        self.vehicle_persistence = 10  # Reduced frames for faster cleanup
        
        # Debug mode
        self.debug = True
    
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
                cv2.putText(annotated_frame, f"Traffic Light: {state}", 
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
        
        cv2.putText(frame, "Green Line", 
                   (self.green_line['x1'], self.green_line['y1'] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        cv2.putText(frame, "Red Line", 
                   (self.red_line['x1'], self.red_line['y1'] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    
    def _check_violations(self, vehicle, frame):
        vehicle_id = vehicle['id']
        center_x, center_y = vehicle['center']
        x1, y1, x2, y2 = vehicle['bbox']
        
        self.vehicle_tracker[vehicle_id]['last_bbox'] = (x1, y1, x2, y2)
        
        if 'green_crossed' not in self.vehicle_tracker[vehicle_id]:
            self.vehicle_tracker[vehicle_id] = {
                'green_crossed': False,
                'red_crossed': False,
                'violation_recorded': False,
                'warning_shown': False,
                'first_seen': datetime.now(),
                'position_history': [],
                'last_bbox': (x1, y1, x2, y2),
                'last_seen': datetime.now()
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
        
        # Update last seen time
        self.vehicle_tracker[vehicle_id]['last_seen'] = datetime.now()
        
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
                           0.6, (0, 0, 255), 1)  # Smaller text
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)  # Thinner box
            
            elif self.vehicle_tracker[vehicle_id]['green_crossed'] and not crossed_red:
                warning_text = "CANH BAO"
                cv2.putText(frame, warning_text, 
                           (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.6, (0, 255, 255), 1)
                self.vehicle_tracker[vehicle_id]['warning_shown'] = True
        
        if vehicle_id in self.active_violations:
            violation_text = "XE VUOT DEN DO"
            cv2.putText(frame, violation_text, 
                       (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (0, 0, 255), 1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
    
    def _clean_vehicle_tracker(self, current_frame_vehicles):
        current_time = datetime.now()
        vehicles_to_remove = set()
        
        for vehicle_id in self.active_violations:
            if vehicle_id not in current_frame_vehicles:
                if vehicle_id in self.vehicle_tracker:
                    last_seen = self.vehicle_tracker[vehicle_id].get('last_seen', current_time)
                    if (current_time - last_seen).total_seconds() > 1:  # Remove after 1 second
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
        cv2.putText(frame, f"Traffic Light:", (20, 35), 
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
    
    def process_video(self, video_path, output_dir='violations', skip_frames=0):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return
        
        if skip_frames > 0:
            for _ in range(skip_frames):
                ret, _ = cap.read()
                if not ret:
                    print("Error skipping frames")
                    return
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_writer = cv2.VideoWriter(str(Path(output_dir) / 'output_video.mp4'),
                                      cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        
        csv_path = Path(output_dir) / 'detections.csv'
        with open(csv_path, mode='w', newline='') as csv_file:
            fieldnames = ['timestamp', 'frame_number', 'class_id', 'class_name', 
                         'x1', 'y1', 'x2', 'y2', 'confidence', 'light_state', 'violation']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            
            frame_count = 0
            last_violation_frame = None
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                processed_frame, vehicles, light_state, has_violations = self.detect_vehicles_and_lights(frame)
                frame_count += 1
                
                for vehicle in vehicles:
                    x1, y1, x2, y2 = vehicle['bbox']
                    vehicle_id = vehicle['id']
                    violation = vehicle_id in self.active_violations
                    
                    writer.writerow({
                        'timestamp': vehicle['timestamp'],
                        'frame_number': frame_count,
                        'class_id': vehicle['class_id'],
                        'class_name': vehicle['class_name'],
                        'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                        'confidence': vehicle['confidence'],
                        'light_state': light_state,
                        'violation': "YES" if violation else "NO"
                    })
                
                video_writer.write(processed_frame)
                
                if has_violations:
                    if last_violation_frame is None or frame_count - last_violation_frame >= int(fps):
                        output_path = Path(output_dir) / f"violation_{frame_count:04d}.jpg"
                        cv2.imwrite(str(output_path), processed_frame)
                        last_violation_frame = frame_count
                
                cv2.imshow('Traffic Violation Detection', processed_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()
        print(f"Processing complete. Detected {self.violation_count} violations. Results saved to {output_dir}")

    def _get_class_color(self, class_id):
        colors = {1: (0, 255, 0), 3: (255, 0, 0), 5: (255, 255, 0), 7: (0, 255, 255)}
        return colors.get(class_id, (128, 128, 128))

if __name__ == "__main__":
    detector = TrafficViolationDetector('yolov8m.pt')
    video_path = 'data/videos/test2.mp4'
    detector.process_video(video_path)