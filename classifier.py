# -*- coding: utf-8 -*-
"""Finds the facial expression of a given image. 
"""
from pathlib import Path

from keras.preprocessing.image import img_to_array
from keras.models import load_model
import cv2
import glob
import numpy as np

# Parameters
face_detection_path = str(Path('data/haarcascade_frontalface_default.xml'))
model_path = str(Path('models/model.best.hdf5'))
input_size = (48, 48)
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Our face detector and trained model
face_detector = cv2.CascadeClassifier(face_detection_path)
emotion_classifier = load_model(model_path, compile=False)

# Gathers a list of all the images in the project directory. 
images = glob.glob('*.jpg') + glob.glob('*.jpeg') + glob.glob('*.png')

# Creating a frame of only the face in an image
if images:
    frame = cv2.imread(images[0])
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.05,
                                            minNeighbors=5, minSize=(30,30))
    faces = sorted(faces, reverse=True,
                   key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
    (x, y, w, h) = faces
    roi = gray_frame[y:y + h, x:x + w]  
    roi = cv2.resize(roi, input_size)
    roi = roi.astype("float") / 255.0
    roi = img_to_array(roi)
    roi = np.expand_dims(roi, axis=0)
    
    # Labels each picture with an expression
    predictions = emotion_classifier.predict(roi)[0]
    expression = emotions[predictions.argmax()]
    cv2.putText(frame, expression, (x, y), 
                cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
    
    cv2.imshow(images[0], frame)      
    cv2.waitKey(0)  
    cv2.destroyAllWindows()  
else:
    print('NO IMAGES FOUND!')
    