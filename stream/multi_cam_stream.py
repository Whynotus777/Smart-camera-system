import cv2
import threading

class CameraStream:
    def __init__(self, cam_id, source):
        self.cam_id = cam_id
        self.cap = cv2.VideoCapture(source)
        self.frame = None
        self.running = True
        threading.Thread(target=self._update, daemon=True).start()

    def _update(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                self.frame = frame

    def read(self):
        return self.frame

    def stop(self):
        self.running = False
        self.cap.release()
