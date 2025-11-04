"""
Camera System with Pose-Based Action Recognition
Optimized for Jetson Nano and RTX 4070
"""

import threading
import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict, deque
import time
import requests
from datetime import datetime
import queue
import json
import os
from pathlib import Path
import torch

# Import pose action detector
from pose_action_detector import PoseActionDetector
from lstm_action_classifier import ActionClassifierTrainer

# Video recorder and ByteTracker
from video_recorder import VideoRecorder
from dataclasses import dataclass

try:
    from byte_tracker_fixed import BYTETracker, STrack
    print("[INFO] Using fixed ByteTracker")
except ImportError:
    from yolox.tracker.byte_tracker import BYTETracker, STrack
    print("[WARNING] Using original ByteTracker")

# ============================================================
# JETSON OPTIMIZATION CONFIGURATION
# ============================================================
JETSON_MODE = True  # Set to False when running on RTX 4070

if JETSON_MODE:
    print("[CONFIG] Running in JETSON MODE - Optimized for Jetson Nano")
    PROCESS_EVERY_N_FRAMES = 5
    YOLO_INPUT_SIZE = 416
    POSE_MODEL_COMPLEXITY = 0
    ENABLE_VIDEO_RECORDING = False
    MAX_CAMERAS = 1  # Start with 1 camera on Jetson Nano
    RESIZE_FOR_YOLO = True
else:
    print("[CONFIG] Running in DESKTOP MODE - Full performance")
    PROCESS_EVERY_N_FRAMES = 2
    YOLO_INPUT_SIZE = 640
    POSE_MODEL_COMPLEXITY = 1
    ENABLE_VIDEO_RECORDING = True
    MAX_CAMERAS = 10
    RESIZE_FOR_YOLO = False
# ============================================================

# Cameras
CAMERAS = {
    "Cam1": "rtsp://admin:IshanKharat@10.229.121.51:554/h264Preview_01_sub",
    #"Cam2": "rtsp://admin:IshanKharat@10.229.121.16:554/h264Preview_01_sub",
}

# Limit cameras in Jetson mode
if JETSON_MODE:
    CAMERAS = dict(list(CAMERAS.items())[:MAX_CAMERAS])

CAMERA_ROLES = {
    "Cam1": "freezer",
}

# Load YOLO - Smart loading for Jetson vs Desktop
print("[INFO] Loading YOLO model...")

# Check if CUDA is available
cuda_available = torch.cuda.is_available()
print(f"[INFO] CUDA Available: {cuda_available}")

if JETSON_MODE and cuda_available:
    # Try to use TensorRT on Jetson with CUDA
    model_path = "yolov8n.engine"
    
    if not os.path.exists(model_path):
        print("[INFO] TensorRT engine not found. Exporting (one-time, ~2-3 minutes)...")
        try:
            temp_model = YOLO("yolov8n.pt")
            temp_model.export(format="engine", half=True, imgsz=416)
            print("[INFO] TensorRT export complete!")
            model = YOLO(model_path)
        except Exception as e:
            print(f"[WARNING] TensorRT export failed: {e}")
            print("[INFO] Falling back to regular YOLO")
            model = YOLO("yolov8n.pt")
    else:
        model = YOLO(model_path)
        print("[INFO] YOLO loaded with TensorRT acceleration")
else:
    # Regular YOLO for CPU or desktop
    model = YOLO("yolov8n.pt")
    print("[INFO] YOLO loaded (CPU mode)" if not cuda_available else "[INFO] YOLO loaded")

# Configuration
CONFIDENCE_THRESHOLD = 0.5
PERSON_CLASS_ID = 0
ALERT_SERVER_URL = "http://localhost:8000/alert"

# Action-based alert configuration
ACTION_SEVERITY_MAP = {
    'reaching': 1,
    'concealing': 3,
    'pocket_gesture': 3,
    'nervous': 2,
    'looking_around': 2,
    'crouching': 2,
}

MIN_ACTION_CONFIDENCE = 0.8

@dataclass
class BYTETrackerArgs:
    track_thresh: float = 0.5
    track_buffer: int = 30
    match_thresh: float = 0.8
    aspect_ratio_thresh: float = 1.6
    min_box_area: float = 10
    mot20: bool = False

# Track history
track_history = defaultdict(lambda: defaultdict(lambda: {
    'first_seen': None,
    'last_seen': None,
    'bbox_history': deque(maxlen=50),
    'zone_history': deque(maxlen=20),
    'actions_detected': [],
    'action_scores': {},
    'alerts': [],
    'alerted': False,
    'suspicious_score': 0,
    'behaviors_detected': set(),
    'pose_detector': None,
}))

# Cross-camera tracking
class CrossCameraTracker:
    def __init__(self):
        self.global_tracks = {}
        self.local_to_global = {}
        self.next_global_id = 1
        self.appearance_threshold = 0.7
    
    def extract_appearance_features(self, frame, bbox):
        x1, y1, x2, y2 = bbox
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(frame.shape[1], x2), min(frame.shape[0], y2)
        
        if x2 <= x1 or y2 <= y1:
            return None
        
        person_crop = frame[y1:y2, x1:x2]
        if person_crop.size == 0:
            return None
        
        hist = cv2.calcHist([person_crop], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        return hist
    
    def compare_appearances(self, feat1, feat2):
        if feat1 is None or feat2 is None:
            return 0.0
        similarity = np.dot(feat1, feat2) / (np.linalg.norm(feat1) * np.linalg.norm(feat2) + 1e-6)
        return similarity
    
    def update(self, camera, local_id, bbox, frame, timestamp):
        key = (camera, local_id)
        appearance = self.extract_appearance_features(frame, bbox)
        
        if key in self.local_to_global:
            global_id = self.local_to_global[key]
            self.global_tracks[global_id]['last_seen'][camera] = timestamp
            self.global_tracks[global_id]['bbox'][camera] = bbox
            return global_id
        
        best_match_id = None
        best_similarity = 0
        
        for global_id, track_data in self.global_tracks.items():
            for other_cam in track_data['cameras']:
                if other_cam == camera:
                    continue
                
                for other_local_id, other_track in track_history[other_cam].items():
                    if 'appearance_vector' in other_track and other_track['appearance_vector'] is not None:
                        similarity = self.compare_appearances(appearance, other_track['appearance_vector'])
                        
                        if similarity > best_similarity and similarity > self.appearance_threshold:
                            best_similarity = similarity
                            best_match_id = global_id
        
        if best_match_id is not None:
            global_id = best_match_id
            self.local_to_global[key] = global_id
            self.global_tracks[global_id]['cameras'].add(camera)
            self.global_tracks[global_id]['last_seen'][camera] = timestamp
            self.global_tracks[global_id]['bbox'][camera] = bbox
        else:
            global_id = self.next_global_id
            self.next_global_id += 1
            self.local_to_global[key] = global_id
            self.global_tracks[global_id] = {
                'cameras': {camera},
                'last_seen': {camera: timestamp},
                'bbox': {camera: bbox}
            }
        
        if appearance is not None:
            track_history[camera][local_id]['appearance_vector'] = appearance
        
        return global_id
    
    def cleanup_old_tracks(self, current_time, timeout=30):
        to_remove = []
        for global_id, track_data in self.global_tracks.items():
            if all(current_time - last_seen > timeout for last_seen in track_data['last_seen'].values()):
                to_remove.append(global_id)
        
        for global_id in to_remove:
            keys_to_remove = [k for k, v in self.local_to_global.items() if v == global_id]
            for key in keys_to_remove:
                del self.local_to_global[key]
            del self.global_tracks[global_id]

cross_camera_tracker = CrossCameraTracker()

# Video recorder - conditional
if ENABLE_VIDEO_RECORDING:
    video_recorder = VideoRecorder(CAMERAS, pre_alert_seconds=10, post_alert_seconds=20)
else:
    video_recorder = None
    print("[INFO] Video recording disabled for Jetson performance")

# LSTM classifier
print("[INFO] Loading LSTM action classifier...")
lstm_classifier = ActionClassifierTrainer(model_path='models/action_lstm.pth')
if lstm_classifier.load_model():
    print("[INFO] ✅ LSTM model loaded successfully!")
else:
    print("[WARNING] ⚠️  LSTM model not found - using pose detection only")
    lstm_classifier = None

# Frame queues
frame_queues = {}
for cam_name in CAMERAS.keys():
    frame_queues[cam_name] = queue.Queue(maxsize=2)

# Alert queue
alert_queue = queue.PriorityQueue()

def load_zones(cam_name):
    filename = f"zones_{cam_name}.json"
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            zones = json.load(f)
            print(f"[{cam_name}] Loaded {len(zones)} zones")
            return zones
    else:
        print(f"[{cam_name}] No zones file found")
        return []

def point_in_polygon(point, polygon):
    x, y = point
    pts = np.array(polygon, np.int32)
    return cv2.pointPolygonTest(pts, (float(x), float(y)), False) >= 0

def get_zone_for_bbox(bbox, zones):
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    
    for zone in zones:
        if point_in_polygon((center_x, center_y), zone['points']):
            return zone['type'], zone.get('alert_threshold', 5)
    
    return None, None

def analyze_pose_actions(track_id, cam_name, bbox, frame, current_time, zones):
    track_info = track_history[cam_name][track_id]
    
    if track_info['first_seen'] is None:
        track_info['first_seen'] = current_time
    
    track_info['last_seen'] = current_time
    track_info['bbox_history'].append(bbox)
    
    zone_type, alert_threshold = get_zone_for_bbox(bbox, zones)
    track_info['zone_history'].append(zone_type)
    
    if track_info['pose_detector'] is None:
        track_info['pose_detector'] = PoseActionDetector(
            history_length=60,
            model_complexity=POSE_MODEL_COMPLEXITY
        )
    
    pose_detector = track_info['pose_detector']
    actions, annotated_crop = pose_detector.process_frame(frame, bbox)
    
    if track_info['alerted']:
        return False, None, 0, actions
    
    suspicious = False
    reason = ""
    severity = 0
    suspicious_actions = []
    
    for action in actions:
        if action.confidence < MIN_ACTION_CONFIDENCE:
            continue
        
        action_severity = ACTION_SEVERITY_MAP.get(action.action_type, 1)
        track_info['actions_detected'].append(action)
        
        if action.action_type not in track_info['action_scores']:
            track_info['action_scores'][action.action_type] = 0
        track_info['action_scores'][action.action_type] += action.confidence
        
        if action_severity == 3:
            suspicious = True
            reason = action.description
            severity = 3
            suspicious_actions.append(action.action_type)
            track_info['behaviors_detected'].add(action.action_type)
    
    if not suspicious and len(actions) >= 2:
        action_types = [a.action_type for a in actions]
        
        if 'looking_around' in action_types and 'concealing' in action_types:
            suspicious = True
            reason = "Looking around while concealing"
            severity = 3
            suspicious_actions = ['looking_around', 'concealing']
    
    # LSTM prediction
    if lstm_classifier is not None and len(pose_detector.pose_history) >= 30:
        pose_sequence = []
        for pose in list(pose_detector.pose_history)[-30:]:
            pose_sequence.append(pose.landmarks)
        
        try:
            predicted_class, confidence = lstm_classifier.predict(pose_sequence)
            
            if confidence > 0.85:
                if predicted_class == 'THEFT':
                    suspicious = True
                    reason = f"ML: Theft sequence ({confidence:.0%})"
                    severity = 3
                    suspicious_actions.append('ml_theft')
                    track_info['behaviors_detected'].add('ml_theft')
        except Exception as e:
            pass
    
    if suspicious and zone_type == 'HIGH_VALUE':
        severity = 3
        reason = f"{reason} in high-value area"
    
    if suspicious:
        track_info['alerted'] = True
        track_info['alerts'].append({
            'time': current_time,
            'reason': reason,
            'severity': severity,
            'actions': suspicious_actions,
            'action_confidences': {a.action_type: a.confidence for a in actions}
        })
    
    return suspicious, reason, severity, actions

def tlwh_to_xyxy(tlwh):
    x, y, w, h = tlwh
    return [x, y, x + w, y + h]

def send_alert(cam_name, track_id, bbox, reason, severity, actions, global_id):
    try:
        payload = {
            "camera": cam_name,
            "track_id": int(track_id),
            "global_id": global_id,
            "bbox": bbox,
            "reason": reason,
            "severity": severity,
            "behaviors": actions,
            "timestamp": datetime.now().isoformat()
        }
        response = requests.post(ALERT_SERVER_URL, json=payload, timeout=0.5)
        if response.status_code == 200:
            print(f"[ALERT-{severity}] {cam_name} - ID:{track_id} - {reason}")
            return True
    except:
        pass
    return False

def alert_worker():
    while True:
        try:
            priority, alert_data = alert_queue.get(timeout=1)
            if alert_data is None:
                break
            
            send_alert(
                alert_data['camera'],
                alert_data['track_id'],
                alert_data['bbox'],
                alert_data['reason'],
                alert_data['severity'],
                alert_data['actions'],
                alert_data['global_id']
            )
            
            if video_recorder:
                video_recorder.start_recording(alert_data)
            
            alert_queue.task_done()
        except queue.Empty:
            continue

def camera_worker(cam_name, rtsp_url):
    print(f"[{cam_name}] Worker starting...")
    
    zones = load_zones(cam_name)
    
    cap = cv2.VideoCapture(rtsp_url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    if not cap.isOpened():
        print(f"[{cam_name}] ERROR: Could not open")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps > 30:
        fps = 10
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"[{cam_name}] Connected: {width}x{height} @ {fps}fps")
    
    tracker_args = BYTETrackerArgs()
    tracker = BYTETracker(tracker_args, frame_rate=fps)
    
    frame_count = 0
    process_every_n_frames = PROCESS_EVERY_N_FRAMES
    failed_reads = 0
    max_failed_reads = 50
    resize_for_yolo = RESIZE_FOR_YOLO
    yolo_size = YOLO_INPUT_SIZE
    
    print(f"[{cam_name}] Processing every {process_every_n_frames} frames, YOLO size: {yolo_size}")
    
    while True:
        ret, frame = cap.read()
        
        if not ret or frame is None:
            failed_reads += 1
            if failed_reads >= max_failed_reads:
                print(f"[{cam_name}] Reconnecting...")
                cap.release()
                time.sleep(1)
                cap = cv2.VideoCapture(rtsp_url)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                failed_reads = 0
            continue
        
        failed_reads = 0
        frame_count += 1
        current_time = time.time()
        
        if video_recorder:
            video_recorder.add_frame(cam_name, frame, current_time)
        
        display_frame = frame.copy()
        
        # Draw zones
        if zones:
            for zone in zones:
                pts = np.array(zone['points'], np.int32)
                pts = pts.reshape((-1, 1, 2))
                color = tuple(zone['color'])
                cv2.polylines(display_frame, [pts], True, color, 2)
        
        # Process every N frames
        if frame_count % process_every_n_frames == 0:
            try:
                if resize_for_yolo:
                    frame_resized = cv2.resize(frame, (yolo_size, yolo_size))
                    results = model(frame_resized, conf=CONFIDENCE_THRESHOLD, classes=[PERSON_CLASS_ID], verbose=False)
                    scale_x = width / yolo_size
                    scale_y = height / yolo_size
                else:
                    results = model(frame, conf=CONFIDENCE_THRESHOLD, classes=[PERSON_CLASS_ID], verbose=False)
                    scale_x = 1.0
                    scale_y = 1.0
                
                online_targets = []
                
                if results[0].boxes is not None and len(results[0].boxes) > 0:
                    detections = []
                    for box in results[0].boxes:
                        xyxy = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf.item())
                        detections.append([
                            xyxy[0] * scale_x,
                            xyxy[1] * scale_y,
                            xyxy[2] * scale_x,
                            xyxy[3] * scale_y,
                            conf
                        ])
                    
                    if detections:
                        detections = np.array(detections)
                        online_targets = tracker.update(detections, [height, width], [height, width])
                else:
                    online_targets = tracker.update(np.empty((0, 5)), [height, width], [height, width])
                
                for track in online_targets:
                    track_id = track.track_id
                    bbox = tlwh_to_xyxy(track.tlwh)
                    bbox = [int(b) for b in bbox]
                    
                    global_id = cross_camera_tracker.update(cam_name, track_id, bbox, frame, current_time)
                    
                    is_suspicious, reason, severity, detected_actions = analyze_pose_actions(
                        track_id, cam_name, bbox, frame, current_time, zones
                    )
                    
                    if is_suspicious:
                        action_names = [a.action_type for a in detected_actions]
                        alert_queue.put((-severity, {
                            'camera': cam_name,
                            'track_id': track_id,
                            'global_id': global_id,
                            'bbox': bbox,
                            'reason': reason,
                            'severity': severity,
                            'actions': action_names
                        }))
                    
                    # Draw
                    x1, y1, x2, y2 = bbox
                    track_info = track_history[cam_name][track_id]
                    
                    if track_info['alerted']:
                        color = (0, 0, 255)
                        label = f"ID:{track_id} ALERT!"
                    else:
                        if detected_actions:
                            highest_severity_action = max(detected_actions,
                                key=lambda a: ACTION_SEVERITY_MAP.get(a.action_type, 1))
                            action_severity = ACTION_SEVERITY_MAP.get(highest_severity_action.action_type, 1)
                            
                            if action_severity == 3:
                                color = (0, 165, 255)
                            elif action_severity == 2:
                                color = (0, 255, 255)
                            else:
                                color = (0, 255, 0)
                            
                            label = f"ID:{track_id} {highest_severity_action.action_type[:8]}"
                        else:
                            color = (0, 255, 0)
                            label = f"ID:{track_id}"
                    
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(display_frame, label, (x1, y1-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                if video_recorder:
                    video_recorder.update_active_recordings(cam_name, display_frame)
                
                if frame_count % 100 == 0:
                    cross_camera_tracker.cleanup_old_tracks(current_time)
                
                info = f"{cam_name} | {datetime.now().strftime('%H:%M:%S')} | People: {len(online_targets)}"
            except Exception as e:
                print(f"[{cam_name}] Error: {e}")
                info = f"{cam_name} | ERROR"
        else:
            if video_recorder:
                video_recorder.update_active_recordings(cam_name, display_frame)
            info = f"{cam_name} | {datetime.now().strftime('%H:%M:%S')}"
        
        cv2.putText(display_frame, info, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        if frame_queues[cam_name].full():
            try:
                frame_queues[cam_name].get_nowait()
            except queue.Empty:
                pass
        
        try:
            frame_queues[cam_name].put(display_frame, block=False)
        except queue.Full:
            pass
    
    cap.release()

def main():
    print("="*60)
    print("SHOPLIFTING DETECTION SYSTEM")
    print("="*60)
    print(f"Mode: {'JETSON' if JETSON_MODE else 'DESKTOP'}")
    print(f"Cameras: {len(CAMERAS)}")
    print(f"Processing: Every {PROCESS_EVERY_N_FRAMES} frames")
    print(f"YOLO Size: {YOLO_INPUT_SIZE}x{YOLO_INPUT_SIZE}")
    print("="*60)
    
    if video_recorder:
        video_recorder.start()
    
    alert_thread = threading.Thread(target=alert_worker, daemon=True)
    alert_thread.start()
    
    worker_threads = []
    for cam_name, rtsp_url in CAMERAS.items():
        t = threading.Thread(target=camera_worker, args=(cam_name, rtsp_url), daemon=True)
        t.start()
        worker_threads.append(t)
        time.sleep(1)
    
    print(f"[INFO] All workers started")
    print("[INFO] Press 'q' to exit")
    
    time.sleep(3)
    
    window_positions = [(0, 50), (660, 50)]
    
    for idx, cam_name in enumerate(CAMERAS.keys()):
        cv2.namedWindow(cam_name, cv2.WINDOW_NORMAL)
        if idx < len(window_positions):
            cv2.moveWindow(cam_name, window_positions[idx][0], window_positions[idx][1])
        cv2.resizeWindow(cam_name, 640, 360)
    
    last_frames = {}
    for cam_name in CAMERAS.keys():
        last_frames[cam_name] = np.zeros((360, 640, 3), dtype=np.uint8)
    
    try:
        while True:
            for cam_name in CAMERAS.keys():
                try:
                    frame = frame_queues[cam_name].get(timeout=0.01)
                    last_frames[cam_name] = frame
                except queue.Empty:
                    pass
                
                cv2.imshow(cam_name, last_frames[cam_name])
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        pass
    
    print("\n[INFO] Shutting down...")
    alert_queue.put((0, None))
    if video_recorder:
        video_recorder.stop()
    cv2.destroyAllWindows()
    time.sleep(1)
    print("[INFO] Done")

if __name__ == "__main__":
    main()
