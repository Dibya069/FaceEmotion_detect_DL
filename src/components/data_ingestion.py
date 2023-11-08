import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt

import os, sys
from src.exception import CustomException
from src.logger import logging
from src.utils import DataIngestionConfig
from dataclasses import dataclass

@dataclass
class Data_inges:
    raw_path = os.path.join("artifacts", "faceEmot.csv")
    # val_apth = os.path.join("artifacts", "validate.npz")
    
    train_faces_path = os.path.join("artifacts", "train_faces.npz")
    train_emotion_path = os.path.join("artifacts", "train_emotions.npz")
    
    test_faces_path = os.path.join("artifacts", "test_faces.npz")
    test_emotion_path = os.path.join("artifacts", "test_emotions.npz")

class dataset_provider:
    def __init__(self, data_ingestion_config: DataIngestionConfig = DataIngestionConfig()):
        try:
            self.data_ingestion_config = data_ingestion_config
            self.data_inges = Data_inges()
        except Exception as e:
            raise CustomException(e, sys)
        
    def pre_process(self, x, _is=True):
        try:
            x = x.astype('float32')
            x = x / 255.0
            if _is:
                x = x - 0.5
                x = x * 2.0
            return x
        except Exception as e:
            logging.info("Error in Preprocess stage")
            raise CustomException(e, sys)


    def dataset_initializer(self):
        try:
            data = pd.read_csv(self.data_ingestion_config.getting_data)
            logging.info('getting data successfully')

            os.makedirs(os.path.dirname(self.data_inges.raw_path), exist_ok=True)
            data.to_csv(self.data_inges.raw_path, index=False, header=True)
            logging.info("Directory and Raw data created successfully")

            pixels = data['pixels'].tolist()
            width = 48
            height = 48
            faces = []

            for pixel_sequence in pixels:
                face = [int(pixel) for pixel in pixel_sequence.split(' ')]
                face = np.asarray(face).reshape(width, height)
                face = cv2.resize(face.astype('uint8'), self.data_ingestion_config.img_size)
                faces.append(face.astype('float32'))
            
            logging.info("Pixels are created for the face")

            faces = np.asarray(faces)
            faces = np.expand_dims(faces, -1)
            faces = self.pre_process(faces)

            emotions = pd.get_dummies(data['emotion']).values
            logging.info("Face pixels and emotions pixels are created")

            n_train = int(0.8 * len(faces))

            ds_train_faces = faces[:n_train]
            ds_train_emotions = emotions[:n_train]

            ds_test_faces = faces[n_train:]
            ds_test_emotions = emotions[n_train:]
            # val_data = (ds_test_faces,ds_test_emotions)

            np.savez(self.data_inges.train_faces_path, data = ds_train_faces)
            np.savez(self.data_inges.train_emotion_path, data = ds_train_emotions)

            np.savez(self.data_inges.test_faces_path, data = ds_test_faces)
            np.savez(self.data_inges.test_emotion_path, data = ds_test_emotions)
            # np.savez(self.data_inges.val_apth, data = val_data, ds_test_faces = val_data[0], ds_test_emotions = val_data[1])
            logging.info("...Saved")


            return(
                self.data_inges.train_faces_path,
                self.data_inges.test_faces_path,
                self.data_inges.train_emotion_path,
                self.data_inges.test_emotion_path
            )
        
        except Exception as e:
            logging.info("Data Initialization Arises...")
            raise CustomException(e, sys)
