from Smoother import Smoother
import cv2
from MyLandmark import MyLandMark
from util import timer


class NoiseRemover:
    def __init__(self, face_mesh, size=10):
        self.face_mesh = face_mesh
        self.image_BGRs = Smoother(size=size)
        self.image_RGBs = Smoother(size=size)
        self.my_landmarks = Smoother(size=size)

    def put(self, image_bgr):
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image_rgb)
        if results.multi_face_landmarks:
            self.image_BGRs.put(image_bgr)
            self.image_RGBs.put(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
            self.my_landmarks.put(MyLandMark(results.multi_face_landmarks[0]))

    def get(self):
        image_BGR = self.image_BGRs.get_median_value()
        image_RGB = self.image_RGBs.get_median_value()
        my_landmark = self.my_landmarks.get_avg(key=lambda x: x.coords)
        return image_BGR, image_RGB, my_landmark
