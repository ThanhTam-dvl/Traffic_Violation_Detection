import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import time
import csv
from datetime import datetime
from collections import deque, defaultdict

class TrafficViolationDetector:
    def __init__(self, model_path='yolov8m.pt'):
        self.model = YOLO(model_path)
        self.vehicle_classes = {1: 'bicycle', 2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
        self.traffic_light_states = {'red': (0, 0, 255), 'yellow': (0, 255, 255), 'green': (0, 255, 0), 'unknown': (128, 128, 128)}
        self.current_light_state = 'unknown'
        
        # Define warning zone and lines (adjusted for better detection)
        self.warning_zone = {
            'x1': 300, 'y1': 400,  # Top-left corner
            'x2': 1000, 'y2': 600   # Bottom-right corner
        }
        
        # Green line: bottom line of warning zone
        self.green_line = {
            'x1': 300, 'y1': 580,
            'x2': 1000, 'y2': 600
        }
        
        # Red line: top line of warning zone (for complete violation)
        self.red_line = {
            'x1': 300, 'y1': 400,
            'x2': 1000, 'y2': 420
        }
        
        # For temporal smoothing of traffic light state
        self.state_history = deque(maxlen=5)
        self.state_history.extend(['unknown'] * 5)
        
        # Track vehicle violations
        self.vehicle_tracker = defaultdict(dict)
        self.violation_count = 0
        self.active_violations = set()  # Track vehicles with complete violations
        self.active_warnings = set()    # Track vehicles with partial violations (crossed green line)
    
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
        
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls)
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x1, y1, x2, y2 = int(x1/scale_factor), int(y1/scale_factor), int(x2/scale_factor), int(y2/scale_factor)
                
                if class_id in self.vehicle_classes:
                    # Calculate vehicle center point (bottom center)
                    center_x = (x1 + x2) // 2
                    center_y = y2  # Bottom of the bounding box
                    
                    # Create unique vehicle ID based on position and class
                    vehicle_id = f"{self.vehicle_classes[class_id]}_{center_x}_{center_y}"
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
                    
                    # Check for violations and draw warnings
                    self._check_violations(vehicle_info, annotated_frame)
                    
                    color = self._get_class_color(class_id)
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    label = f"{self.vehicle_classes[class_id]} {conf:.2f}"
                    cv2.putText(annotated_frame, label, (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                elif class_id == 9 and conf > 0.3:
                    traffic_light_roi = frame[y1:y2, x1:x2]
                    light_state = self._detect_light_state(traffic_light_roi)
                    area = (x2 - x1) * (y2 - y1)
                    traffic_lights.append({
                        'bbox': (x1, y1, x2, y2),
                        'state': light_state,
                        'confidence': conf,
                        'area': area
                    })
        
        # Update active violations and warnings: only keep vehicles that are still in the frame
        self.active_violations = {vid for vid in self.active_violations if vid in current_frame_vehicles}
        self.active_warnings = {vid for vid in self.active_warnings if vid in current_frame_vehicles}
        
        # Process traffic lights
        if traffic_lights:
            if len(traffic_lights) > 1:
                traffic_lights.sort(key=lambda x: x['area'], reverse=True)
                closest_light = traffic_lights[0]
            else:
                closest_light = traffic_lights[0]
            
            new_state = closest_light['state']
            self.state_history.append(new_state)
            smoothed_state = self._smooth_light_state()
            self.current_light_state = smoothed_state
            print(f"Traffic Light State: {self.current_light_state}")
            
            for tl in traffic_lights:
                x1, y1, x2, y2 = tl['bbox']
                state = tl['state']
                state_color = self.traffic_light_states[state]
                thickness = 4 if tl == closest_light else 2
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), state_color, thickness)
                cv2.putText(annotated_frame, f"Traffic Light: {state}", 
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, state_color, 2)
        else:
            self.state_history.append('unknown')
            self.current_light_state = self._smooth_light_state()
            print("No traffic lights detected, state set to unknown")
        
        self._display_status_info(annotated_frame)
        return annotated_frame, vehicle_boxes, self.current_light_state
    
    def _draw_detection_zones(self, frame):
        """Draw the warning zone and detection lines on the frame."""
        # Draw warning zone (semi-transparent)
        zone_overlay = frame.copy()
        cv2.rectangle(zone_overlay, 
                     (self.warning_zone['x1'], self.warning_zone['y1']),
                     (self.warning_zone['x2'], self.warning_zone['y2']),
                     (0, 165, 255), -1)  # Orange color
        cv2.addWeighted(zone_overlay, 0.3, frame, 0.7, 0, frame)
        
        # Draw green line (bottom line)
        cv2.line(frame, 
                (self.green_line['x1'], self.green_line['y1']),
                (self.green_line['x2'], self.green_line['y2']),
                (0, 255, 0), 2)  # Green color
        
        # Draw red line (top line)
        cv2.line(frame, 
                (self.red_line['x1'], self.red_line['y1']),
                (self.red_line['x2'], self.red_line['y2']),
                (0, 0, 255), 2)  # Red color
        
        # Label the lines
        cv2.putText(frame, "Green Line", 
                   (self.green_line['x1'], self.green_line['y1'] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, "Red Line", 
                   (self.red_line['x1'], self.red_line['y1'] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    def _check_violations(self, vehicle, frame):
        """Check if a vehicle is violating traffic rules and draw warnings."""
        vehicle_id = vehicle['id']
        center_x, center_y = vehicle['center']
        
        # Check if vehicle is in warning zone
        in_warning_zone = (self.warning_zone['x1'] <= center_x <= self.warning_zone['x2'] and
                          self.warning_zone['y1'] <= center_y <= self.warning_zone['y2'])
        
        # Check if vehicle has crossed the green line
        crossed_green = center_y >= self.green_line['y1']
        
        # Check if vehicle has crossed the red line (complete violation)
        crossed_red = center_y <= self.red_line['y2']
        
        # Debug crossing status
        print(f"Vehicle {vehicle_id}: center_y={center_y}, crossed_green={crossed_green} (green_line_y1={self.green_line['y1']}), crossed_red={crossed_red} (red_line_y2={self.red_line['y2']})")
        
        # Initialize tracking for new vehicles
        if vehicle_id not in self.vehicle_tracker:
            self.vehicle_tracker[vehicle_id] = {
                'green_crossed': False,
                'red_crossed': False,
                'violation_recorded': False,
                'warning_shown': False
            }
        
        # Update crossing status
        if crossed_green:
            self.vehicle_tracker[vehicle_id]['green_crossed'] = True
            print(f"Vehicle {vehicle_id} crossed green line")
        
        if crossed_red:
            self.vehicle_tracker[vehicle_id]['red_crossed'] = True
            print(f"Vehicle {vehicle_id} crossed red line")
        
        # Check for violation (only when light is red)
        if self.current_light_state == 'red':
            # Partial violation: crossed green line
            if self.vehicle_tracker[vehicle_id]['green_crossed'] and not self.vehicle_tracker[vehicle_id]['red_crossed']:
                if not self.vehicle_tracker[vehicle_id]['warning_shown']:
                    self.active_warnings.add(vehicle_id)
                    self.vehicle_tracker[vehicle_id]['warning_shown'] = True
                    print(f"Warning issued for {vehicle_id}: Approaching Red Light")
            
            # Complete violation: crossed red line
            if self.vehicle_tracker[vehicle_id]['red_crossed']:
                if not self.vehicle_tracker[vehicle_id]['violation_recorded']:
                    self.vehicle_tracker[vehicle_id]['violation_recorded'] = True
                    self.violation_count += 1
                    self.active_violations.add(vehicle_id)
                    # Remove from active_warnings since it's now a complete violation
                    self.active_warnings.discard(vehicle_id)
                    print(f"Violation recorded for {vehicle_id}, Total Violations: {self.violation_count}")
        
        # Draw warnings for partial violations (yellow)
        if vehicle_id in self.active_warnings:
            x1, y1, x2, y2 = vehicle['bbox']
            cv2.putText(frame, "WARNING: Approaching Red Light", 
                       (x1, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (0, 255, 255), 2)  # Yellow color
            # Draw yellow border around vehicle
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
        
        # Draw warnings for complete violations (red)
        if vehicle_id in self.active_violations:
            x1, y1, x2, y2 = vehicle['bbox']
            cv2.putText(frame, "VIOLATION: Red Light Running", 
                       (x1, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (0, 0, 255), 2)  # Red color
            # Draw red border around violating vehicle
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
    
    def _display_status_info(self, frame):
        """Display traffic light and violation info in top-left corner."""
        # Draw background rectangle for status info
        cv2.rectangle(frame, (10, 10), (300, 120), (0, 0, 0), -1)
        
        # Display traffic light state
        light_color = self.traffic_light_states[self.current_light_state]
        cv2.putText(frame, f"Traffic Light:", (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"{self.current_light_state.upper()}", (20, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, light_color, 3)
        cv2.circle(frame, (200, 50), 15, light_color, -1)
        cv2.circle(frame, (200, 50), 17, (255, 255, 255), 2)
        
        # Display violation count below traffic light info
        cv2.putText(frame, f"Total Violations:", (20, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"{self.violation_count}", (200, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
    
    def _smooth_light_state(self):
        """Apply temporal smoothing to avoid flickering of traffic light state."""
        state_counts = {'red': 0, 'yellow': 0, 'green': 0, 'unknown': 0}
        for state in self.state_history:
            state_counts[state] += 1
        return max(state_counts, key=state_counts.get)
    
    def _detect_light_state(self, traffic_light_roi):
        if traffic_light_roi.size == 0:
            return 'unknown'
        
        gray = cv2.cvtColor(traffic_light_roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        (min_val, max_val, min_loc, max_loc) = cv2.minMaxLoc(blurred)
        center_x, center_y = max_loc
        
        radius = 10
        x1 = max(0, center_x - radius)
        y1 = max(0, center_y - radius)
        x2 = min(traffic_light_roi.shape[1], center_x + radius)
        y2 = min(traffic_light_roi.shape[0], center_y + radius)
        
        focused_roi = traffic_light_roi[y1:y2, x1:x2]
        if focused_roi.size == 0:
            return 'unknown'
        
        hsv = cv2.cvtColor(focused_roi, cv2.COLOR_BGR2HSV)
        red_lower1 = np.array([0, 120, 120])
        red_upper1 = np.array([15, 255, 255])
        red_lower2 = np.array([165, 120, 120])
        red_upper2 = np.array([180, 255, 255])
        green_lower = np.array([40, 50, 50])
        green_upper = np.array([90, 255, 255])
        yellow_lower = np.array([20, 100, 100])
        yellow_upper = np.array([30, 255, 255])
        
        red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
        red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        green_mask = cv2.inRange(hsv, green_lower, green_upper)
        yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
        
        red_pixels = cv2.countNonZero(red_mask)
        green_pixels = cv2.countNonZero(green_mask)
        yellow_pixels = cv2.countNonZero(yellow_mask)
        
        max_pixels = max(red_pixels, green_pixels, yellow_pixels)
        if max_pixels < 5:
            return 'unknown'
        elif max_pixels == red_pixels:
            return 'red'
        elif max_pixels == green_pixels:
            return 'green'
        elif max_pixels == yellow_pixels:
            return 'yellow'
        else:
            return 'unknown'
    
    def process_video(self, video_path, output_dir='violations'):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return
        
        csv_path = Path(output_dir) / 'detections.csv'
        with open(csv_path, mode='w', newline='') as csv_file:
            fieldnames = ['timestamp', 'frame_number', 'class_id', 'class_name', 
                         'x1', 'y1', 'x2', 'y2', 'confidence', 'light_state', 'violation']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            
            frame_count = 0
            previous_violation_count = 0  # Track previous violation count to detect new violations
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                processed_frame, vehicles, light_state = self.detect_vehicles_and_lights(frame)
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
                
                # Save frame only when a new violation occurs
                if self.violation_count > previous_violation_count:
                    output_path = Path(output_dir) / f"violation_{frame_count:04d}.jpg"
                    cv2.imwrite(str(output_path), processed_frame)
                    print(f"Saved violation frame at {output_path}")
                    previous_violation_count = self.violation_count
                
                cv2.imshow('Traffic Violation Detection', processed_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        cap.release()
        cv2.destroyAllWindows()
        print(f"Processing complete. Results saved to {output_dir}")

    def _get_class_color(self, class_id):
        colors = {1: (0, 255, 0), 2: (0, 0, 255), 3: (255, 0, 0), 5: (255, 255, 0), 7: (0, 255, 255)}
        return colors.get(class_id, (128, 128, 128))

if __name__ == "__main__":
    detector = TrafficViolationDetector('yolov8m.pt')
    video_path = 'data/videos/test2.mp4'
    detector.process_video(video_path)