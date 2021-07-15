from gl_wraps.FaceMesh import *
from smooth_filters.NoiseRemover import NoiseRemover
from utils.EventFrequencyCounter import EventFrequencyCounter
from utils.util import *


face_mesh_gl = FaceMesh(cv2.imread('pict/trump.jpg'))

fps_counter = EventFrequencyCounter(print_periodic_reports=True)
cap = cv2.VideoCapture(0)
window = init_gl(shape=(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

with mp.solutions.face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
    noise_remover = NoiseRemover(face_mesh, size=5)
    while cap.isOpened():
        success, image_BGR = cap.read()
        # Сразу откажаем вокруг вертикальной оси
        image_BGR = cv2.flip(image_BGR, 1)
        noise_remover.put(image_BGR)
        image_BGR, image_RGB, my_landmark = noise_remover.get()
        face_mesh_gl.landmark_update_vertex_coords(my_landmark).draw()
        image_BGR_with_replacement = update_image_with_opengl(image_BGR, window)
        plot(image_BGR_with_replacement, wait=False)
        end_frame(window)
        fps_counter.event_occurrence()

glfw.terminate()
