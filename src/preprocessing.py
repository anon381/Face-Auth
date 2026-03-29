import cv2
import numpy as np
import os

IMAGE_SIZE = 64
HAAR_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)

if face_cascade.empty():
    raise IOError("Failed to load Haar cascade. Check OpenCV installation.")

def detect_and_crop_face(image):
    if image is None:
        return None
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    if len(faces) == 0:
        return None
    
    # Choose the biggest face (by area)
    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    (x, y, w, h) = faces[0]
    
    # Add a small margin to avoid cutting too tight
    margin = int(0.1 * min(w, h))
    x = max(0, x - margin)
    y = max(0, y - margin)
    w = min(image.shape[1] - x, w + 2 * margin)
    h = min(image.shape[0] - y, h + 2 * margin)
    
    cropped = image[y:y+h, x:x+w]
    return cropped


def preprocess_face(cropped_face):
    if cropped_face is None:
        return None
    resized = cv2.resize(cropped_face, (IMAGE_SIZE, IMAGE_SIZE))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    normalized = gray.astype(np.float32) / 255.0
    
    return normalized