from OpenGL.GL import *
import glfw
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np
import cv2


class Mesh:
    def __init__(self, face_indexes, cv_image, cv_vertex_coords_on_img):
        """

        :param face_indexes: np.array with shape=(n, 3)
        :param cv_image: result of cv2.imread('...')
        :param cv_vertex_coords_on_img: np.array with shape=(n, 2)
        """
        self.face_indexes = face_indexes.reshape(-1).astype(np.uint32)
        self.image = cv_image
        self.image_rgba = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGBA)
        self.img_data = self.image_rgba.tobytes()
        self.vertex_coords_on_img = cv_vertex_coords_on_img
        self.cur_vertex_coords = np.array([])
        self.debug = True

    def draw(self):
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, self.image_rgba.shape[1], self.image_rgba.shape[0], 0, GL_RGBA, GL_UNSIGNED_BYTE, self.img_data)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, glGenBuffers(1))
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.face_indexes.nbytes, self.face_indexes, GL_STATIC_DRAW)

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        points = np.concatenate([self.cur_vertex_coords, self.vertex_coords_on_img], axis=1).reshape(-1).astype(np.float32)
        glBufferData(GL_ARRAY_BUFFER, points.nbytes, points, GL_STATIC_DRAW)
        glDrawElements(GL_TRIANGLES, len(self.face_indexes), GL_UNSIGNED_INT, None)

    def update_vertex_coords(self, coords):
        self.cur_vertex_coords = coords
        return self
