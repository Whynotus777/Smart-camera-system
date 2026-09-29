import os
import cv2

rtsp_url = os.environ["SCS_CAM1_RTSP_URL"]  # see .env.example
cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    raise RuntimeError("[ERROR] Cannot open RTSP stream")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    cv2.imshow("Camera Stream", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

