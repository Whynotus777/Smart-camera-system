##Authors:
## Ishan Kharat: Driver
## Abdul Manan: Navigator

import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import os

# ------------------- CONFIG -------------------
RTSP_URL = "rtsp://admin:IshanKharat@10.229.121.51:554/h264Preview_01_sub"
OUTPUT_VIDEO = "shoplifting_output.avi"
OUTPUT_FPS = 20.0
SAVE_FRAMES = True
FRAME_DIR = "output_frames"
# Define your shelf region manually (x1, y1, x2, y2)
SHELF_REGION = (50, 50, 590, 300)  
# ----------------------------------------------

yolo_model = YOLO("yolov8n.pt")  # lightweight for Jetson
tracker = DeepSort(max_age=30)

cap = cv2.VideoCapture(RTSP_URL)
if not cap.isOpened():
    raise RuntimeError("[ERROR] Cannot open RTSP stream")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter(
    OUTPUT_VIDEO,
    cv2.VideoWriter_fourcc(*"XVID"),
    OUTPUT_FPS,
    (width, height),
)

if SAVE_FRAMES:
    os.makedirs(FRAME_DIR, exist_ok=True)
frame_count = 0

print("[INFO] Starting shoplifting PoC with YOLO + DeepSORT + Shoplifting logic...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[WARN] Failed to grab frame")
        break

    # Draw shelf region
    x1_s, y1_s, x2_s, y2_s = SHELF_REGION
    cv2.rectangle(frame, (x1_s, y1_s), (x2_s, y2_s), (255, 255, 0), 2)

    # YOLO detection
    results = yolo_model(frame)[0]
    detections = []
    item_boxes = []

    for i, box in enumerate(results.boxes.xyxy):
        x1, y1, x2, y2 = map(int, box)
        conf = float(results.boxes.conf[i])
        cls = int(results.boxes.cls[i])
        # Person and items
        if cls in [0]:  # person
            detections.append(([x1, y1, x2, y2], conf, cls))
        else:  # items like bottle, cellphone, etc
            item_boxes.append((x1, y1, x2, y2, cls))

    # Update tracker
    tracks = tracker.update_tracks(detections, frame=frame)

    # Check shoplifting: person interacts with item outside shelf
    for track in tracks:
        if not track.is_confirmed():
            continue
        tx1, ty1, tx2, ty2 = map(int, track.to_ltrb())
        person_id = track.track_id
        for (ix1, iy1, ix2, iy2, cls) in item_boxes:
            # Check overlap with person
            overlap_x = max(0, min(tx2, ix2) - max(tx1, ix1))
            overlap_y = max(0, min(ty2, iy2) - max(ty1, iy1))
            if overlap_x > 0 and overlap_y > 0:
                # Check if item is outside shelf
                if ix1 < x1_s or ix2 > x2_s or iy1 < y1_s or iy2 > y2_s:
                    alert_text = f"[ALERT] Person ID {person_id} possibly shoplifting!"
                    print(alert_text)
                    cv2.putText(frame, alert_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Draw tracks
    for track in tracks:
        if not track.is_confirmed():
            continue
        bbox = track.to_ltrb()
        tid = track.track_id
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID {tid}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Draw items
    for (ix1, iy1, ix2, iy2, cls) in item_boxes:
        cv2.rectangle(frame, (ix1, iy1), (ix2, iy2), (0, 0, 255), 2)

    out.write(frame)
    if SAVE_FRAMES:
        cv2.imwrite(f"{FRAME_DIR}/frame_{frame_count:05d}.jpg", frame)
        frame_count += 1

    cv2.imshow("Shoplifting Demo", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

