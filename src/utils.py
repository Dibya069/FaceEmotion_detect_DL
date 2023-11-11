import os, sys
from src.exception import CustomException
from src.logger import logging
from src.entity import *
from dataclasses import dataclass
from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
import cv2

import numpy as np
import pandas as pd

@dataclass
class DataIngestionConfig:

    img_size = IMG_SIZE
    getting_data: str = GET_DATA


@dataclass
class ModelCreation:

    input_shape = INPUT_SHAPE
    class_no: int = CLASS_NO
    batch_no: int = BATCH_NO
    #callbacks
    early_stop = EarlyStopping('val_loss', patience=50)
    reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1, patience=int(50/4), verbose=1)
    callbacks = [early_stop, reduce_lr]
    # data generator
    data_generator = ImageDataGenerator(
                            featurewise_center=False,
                            featurewise_std_normalization=False,
                            rotation_range=10,
                            width_shift_range=0.1,
                            height_shift_range=0.1,
                            zoom_range=.1,
                            shear_range =10,
                            horizontal_flip=True)
    

@dataclass
class FaceDetec:

    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    demo_img = DEMO_IMG
    demo_path = DEMO_PATH


@dataclass
class ModelLoad:

    jsonM = JSON_MODEL
    orgM = MODEL
    emolib = EMOTION_LIB
