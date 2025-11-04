import cv2

rtsp_url = "rtsp://admin:IshanKharat@10.229.121.51:554/h264Preview_01_sub"
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

