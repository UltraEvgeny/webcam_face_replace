import numpy as np
import mediapipe as mp


class MyLandMark:
    def __init__(self, multi_face_landmark):
        self.coords = np.array([[lm.x, lm.y, lm.z] for lm in multi_face_landmark.landmark])
