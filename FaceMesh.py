import cv2
import mediapipe as mp
import pyvista
from triangles import TexturedTriangle
from util import *
from Mesh import Mesh
from MyLandmark import MyLandMark


class FaceMesh(Mesh):
    def __init__(self, face_img):
        face_mesh = mp.solutions.face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        initial_multi_face_landmark = MyLandMark(face_mesh.process(face_img).multi_face_landmarks[0])
        face_indices = FaceMesh.get_faces_from_multi_face_landmark(initial_multi_face_landmark)
        init_mesh_points_2d = initial_multi_face_landmark.coords[:, :2]
        super().__init__(face_indices, face_img, init_mesh_points_2d)

    def landmark_update_vertex_coords(self, multi_face_landmark):
        r = multi_face_landmark
        r[:, 1] = 1 - r[:, 1]
        r[:, :2] = r[:, :2] * 2 - 1
        self.update_vertex_coords(r)
        return self

    @staticmethod
    def get_faces_from_multi_face_landmark(multi_face_landmark):
        mesh = pyvista.PolyData(multi_face_landmark.coords).delaunay_2d()
        return mesh.faces.reshape(-1, 4)[:, 1:]
