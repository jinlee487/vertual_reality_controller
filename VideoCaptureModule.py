import cv2

class VCModule():

    def __init__(self, camera_port=0):
        self.camera_port = camera_port

    def getVideoCapture(self):
        return cv2.VideoCapture(self.camera_port,cv2.CAP_DSHOW)