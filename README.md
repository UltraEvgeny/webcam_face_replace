### Webcam face replacement with FaceMesh and OpenGL

How it works:
1) Create a mesh from a photo.
2) Get piture from webcamera, create mesh from it and retexture it according to mesh from (1)
3) Draw this model using opengl and replace your face with created model.
4) Do some color correction and smoothing, because FaceMesh meshes are twitchy.
5) Repeat it. About 20 times per second on my notebook.
 
How to run: main.py
Example of faceswapping: trump_example.jpg
