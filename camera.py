import cv2

class Camera:
    def __init__(self, ip):
        if ip != "0":
            self.url = "rtsp://admin:admin@"+ip+"/1"
        else:
            self.url = 0
        self.cap = None
    
    def open(self):
        try:
            self.cap = cv2.VideoCapture(self.url)
            if not self.cap.isOpened():
                print("Error opening camera")
                raise Exception("Error with camera")

        except Exception as e:
            self.cap = None
            raise Exception("Error with camera")
        
    def readFrame(self):
        try:
            ret, frame = self.cap.read()
            if not ret:
                print("Error reading frame")
                raise Exception("Error with camera")
            else:
                return frame
        except Exception as e:
            print(e)
            raise Exception("Error with camera")
        
    def isOpened(self):
        return self.cap.isOpened()
            
        
    def close(self):
        if self.cap:
            try:
                self.cap.release()
            except Exception as e:
                print("Error closing camera")
                raise Exception("Error with camera")