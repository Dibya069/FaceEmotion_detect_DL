from keras.preprocessing import image
#from tensorflow.keras.preprocessing.image import load_img
import cv2
import pandas as pd
import numpy as np
import os, sys

from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
from src.utils import FaceDetec

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
                cropped_img = np.expand_dims(np.expand_dims(cv2.resize(gray_frame, (48, 48)), -1), 0)
                cropped.append(cropped_img)
                xs.append(x)
                ys.append(y)

            logging.info("All required things created successfuly")

            return xs, ys, img, cropped
        
        except Exception as e:
            raise CustomException(e, sys)
