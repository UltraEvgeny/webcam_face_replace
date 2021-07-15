from contextlib import contextmanager
from datetime import datetime
import time
from OpenGL.GL import *
import glfw
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np
import cv2
from skimage.exposure import match_histograms
import pandas as pd


@contextmanager
def timer(text):
    start_time = time.time_ns()
    try:
        yield
    finally:
        end_time = time.time_ns()
        print(f'Seconds spent on {text}: {(end_time-start_time)/1_000_000:.3f} miliseconds')


def plot(img, name='image', wait=True):
    cv2.imshow(name, img)
    if wait:
        cv2.waitKey(0)


def init_gl(shape):
    float_size = np.dtype(np.float32).itemsize
    vertex_src = """
    # version 330
    layout(location = 0) in vec3 a_position;
    layout(location = 1) in vec2 a_texture;
    out vec2 v_texture;
    void main()
    {
        gl_Position = vec4(a_position, 1.0);
        v_texture = a_texture;
    }
    """

    fragment_src = """
    # version 330
    in vec2 v_texture;
    out vec4 out_color;
    uniform sampler2D s_texture;
    void main()
    {
        out_color = texture(s_texture, v_texture); // * vec4(v_color, 1.0f);
    }
    """

    # initializing glfw library
    if not glfw.init():
        raise Exception("glfw can not be initialized!")

    # creating the window

    window = glfw.create_window(*shape, "My OpenGL window", None, None)
    window.shape = shape
    glfw.hide_window(window)

    # check if window was created
    if not window:
        glfw.terminate()
        raise Exception("glfw window can not be created!")

    # set window's position
    glfw.set_window_pos(window, 400, 200)

    # make the context current
    glfw.make_context_current(window)

    shader = compileProgram(compileShader(vertex_src, GL_VERTEX_SHADER),
                            compileShader(fragment_src, GL_FRAGMENT_SHADER))

    glUseProgram(shader)
    glClearColor(0.0, 0, 0, 0)

    VBO = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, VBO)

    glEnable(GL_DEPTH_TEST)

    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, float_size * 5, ctypes.c_void_p(0))

    glEnableVertexAttribArray(1)
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, float_size * 5, ctypes.c_void_p(float_size * 3))

    texture = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture)
    return window


def end_frame(window):
    glfw.poll_events()
    glfw.swap_buffers(window)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)



def match_histograms_optimized(image, reference, multichannel=False):
    for c in range(3):
        cur_mean = image[:, :, c].mean()
        ref_mean = reference[:, :, c].mean()
        if cur_mean > 0:
            image[:, :, c] = image[:, :, c]*ref_mean/cur_mean
    # cur_mean = image.mean()
    # ref_mean = reference.mean()
    # if cur_mean > 0:
    #     image = image / cur_mean * ref_mean

    #image = match_histograms(image, reference, multichannel=multichannel)
    return image


def update_image_with_opengl(image_bgr, window):
    image_bgr = image_bgr/255
    cv2_img = glReadPixels(0, 0, *window.shape, GL_BGRA, GL_FLOAT)
    cv2_img = cv2_img.reshape(cv2_img.shape[1], cv2_img.shape[0], cv2_img.shape[2])
    cv2_img = cv2.flip(cv2_img, flipCode=0)
    mask = cv2_img[:, :, 3].astype('uint8')
    image_bgr_face = cv2.bitwise_or(image_bgr, image_bgr, mask=mask)
    image_bgr_no_face = cv2.bitwise_or(image_bgr, image_bgr, mask=1-mask)
    cv2_img_matched = match_histograms_optimized(cv2_img[:, :, :3], image_bgr_face, multichannel=True)
    new_array = image_bgr_no_face + cv2_img_matched
    return new_array


