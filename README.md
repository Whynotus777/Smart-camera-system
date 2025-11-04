# 10-Camera-Shoplifting-Detection
This system uses state-of-the-art computer vision and machine learning to detect shoplifting in real-time. It combines multiple AI technologies:

- YOLOv8 for person detection
- ByteTracker for multi-object tracking
- MediaPipe for pose estimation
- LSTM Neural Networks for temporal-action recognition

### Authors:

    Name: Ishan Kharat (ishanmk@umd.edu)   Driver

    Name: Abdul Manan (abdul@quantumroboticslab.com)  Navigator


### High-Level Flow:

        RTSP Camera Streams
            ↓
       YOLOv8 Detection (Find people)
            ↓
       ByteTracker (Track each person)
            ↓
       MediaPipe Pose (Detect body keypoints)
            ↓
       Action Recognition (Analyze gestures)
            ↓
       LSTM Classifier (Understand sequences)
            ↓
       Alert System (Notify if suspicious)
            ↓
       Video Recording (Save evidence)

### Technical Details:
    Architecture: CSPDarknet53 backbone
    Parameters: ~3.2 million (YOLOv8n - nano version)
    Input size: 640x360 (optimized for speed)
    Output: [x1, y1, x2, y2, confidence, class]
    Speed: 100+ FPS on RTX 4070, 25-30 FPS on Jetson
    Accuracy: 95%+ person detection accuracy


### System Integration Flow:

    1. RTSP Stream → Receive frame (30ms)
       ↓
    2. YOLOv8 → Detect people (30ms)
       ↓
    3. ByteTracker → Assign/update track IDs (5ms)
       ↓
    4. For each tracked person:
       |
       ├→ MediaPipe → Extract 33 keypoints (15ms)
       |   ↓
       ├→ Action Recognition → Analyze gestures (5ms)
       |   ↓
       └→ Every 30 frames:
           LSTM → Predict sequence class (10ms)
           ↓
           If suspicious:
           |
           ├→ Generate alert
           ├→ Start video recording
           ├→ Send to dashboard
           └→ SMS notification

Total latency: ~100ms per person


## Quick Start


    git clone https://github.com/IshanMahesh/10-Camera-Shoplifting-Detection_Quantum.git

    python -m venv .venv && source .venv/bin/activate

    pip install --upgrade pip

    pip install -r requirements.txt

Edit camera sources in `deepsort_poc.py` to use your RTSP URLs:

ex: where admin is your camera username and IshanKharat should be your password (change the username and password accordingly from the app)
```python
RTSP_URL = "rtsp://admin:IshanKharat@10.229.121.51:554/bcs/channel0_sub.bcs"
```

Make sure your device is reading the camera. To test it run:

```bash
python test.py
```
Once your camera is running successfully, it means your device reads the camera data.


Now run the code
```bash
python camera_system_with_lstm.py
```

Notes:
- IOU fallback is fine for PoC but not production.
- For 10 cams, use GPU + hardware decoding (FFmpeg/GStreamer).



#### Directory Structure

```bash
10-Camera-Shoplifting-Detection_Quantum/
│
├── alerts/
│   ├── notifier.py
| 
├── spills/
│   ├── spill_detector.py
│
├── stream/
│   ├── multi_cam_stream.py
│
├── alerts/
│   ├── notifier.py
│
├── camera_system_with_lstm.py                 # Main runner
├── requirements.txt
├── testcamera.py
├── yolo11n.pt
└── README.md
