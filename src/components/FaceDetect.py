from keras.preprocessing import image
#from tensorflow.keras.preprocessing.image import load_img
import cv2
import pandas as pd
import numpy as np
import os, sys
import matplotlib.pyplot as plt

from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
from src.utils import FaceDetec, ModelLoad
from src.components.ModelLoad import LoadModel


## Model Load
obj3 = LoadModel()
loaded_model = obj3.model_initiate()
loaded_model.load_weights(ModelLoad.orgM)  # load weights into new model

@dataclass
class FacetDetect:
    def __init__(self, face_detect: FaceDetec = FaceDetec()):
        self.face_detec = face_detect

    def detect_face(self, pic):
        try:
            pic = cv2.imread(pic)
            img = pic.copy()
            img = cv2.resize(img, (600,500))
            #convert image into gray scale as opencv face detector expects gray images
            logging.info("convert image into gray scale as opencv face detector expects gray images")

            gray_image=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            #use haar classifier to detect faces
            logging.info("use haar classifier to detect faces")

            face_box = self.face_detec.cascade.detectMultiScale(gray_image, scaleFactor = 1.3, minNeighbors = 5)
            cropped=[]
            xs=[]
            ys=[]

            logging.info("crop the boxed face")
            for(x,y,w,h) in face_box:
                cv2.rectangle(img, (x,y), (x+w, y+h+10), (0,255,0), 2)
                #crop the boxed face

                gray_frame = gray_image[y:y+h, x:x+w]         # Extract the region of interest (ROI), which is the grayscale face area.
                cropped_img = np.expand_dims(np.expand_dims(cv2.resize(gray_frame, (64, 64)), -1), 0)
                cropped.append(cropped_img)
                xs.append(x)
                ys.append(y)

            logging.info("All required things created successfuly")

            return xs, ys, img, cropped
        
        except Exception as e:
            raise CustomException(e, sys)
        

    def process_detected_faces(self, face_box, frame):
        xs = []
        ys = []
        cropped = []

        for (x, y, w, h) in face_box:

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_face = gray_frame[y:y+h, x:x+w]

            resized_face = cv2.resize(gray_face, (64, 64))
            # Normalize pixel values to be in the range [0, 1]
            normalized_face = resized_face / 255.0
            # Add a channel dimension to the image (assuming your model expects shape (64, 64, 1))
            normalized_face = np.expand_dims(normalized_face, axis=-1)
            normalized_face = normalized_face.reshape((1, 64, 64, 1))
            cropped.append(normalized_face)

            xs.append(x)
            ys.append(y)

        return xs, ys, cropped

        

    def web_cam(self, source):
        try:
            cap = cv2.VideoCapture(source)  # 0 for the default webcam

            fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Define the codec
            out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))  # Create the video writer object


            while True:
                # Capture frame-by-frame
                ret, frame = cap.read()

                # Check if the frame was captured successfully
                if not ret:
                    print("Failed to capture frame from the webcam.")
                    break                
                print("Frame shape:", frame.shape)

                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                print("Grey_Frame shape:", gray_frame.shape)
                
                # Use haar classifier to detect faces
                face_box = self.face_detec.cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)
                xs, ys, cropped = self.process_detected_faces(face_box, frame)

                # Draw rectangles around detected faces
                for (x, y, w, h) in face_box:
                    cv2.rectangle(frame, (x, y), (x+w, y+h+10), (0, 255, 0), 2)

                out.write(frame)

                j=0
                for i in cropped:
                    emotion_pred = int(np.argmax(loaded_model.predict(i)))
                    cv2.putText (frame, ModelLoad.emolib[emotion_pred], (xs[j], ys[j]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    out.write(frame)
                    j+=1

                cv2.imshow('Webcam Face Detection', frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # Release the webcam and close all windows when finished
            out.release()
            cap.release()
            cv2.destroyAllWindows()
        
        except Exception as e:
            raise CustomException(e, sys)
