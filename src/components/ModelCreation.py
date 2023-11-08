import os, sys
import pandas as pd
import numpy as np

from src.exception import CustomException
from src.logger import logging
from src.utils import ModelCreation
from dataclasses import dataclass

"""from keras.layers import Activation, Convolution2D, Dropout, Conv2D
from keras.layers import AveragePooling2D, BatchNormalization
from keras.layers import GlobalAveragePooling2D
from keras.models import Sequential
from keras.layers import Flatten
from keras.models import Model
from keras.layers import Input
from keras.layers import MaxPooling2D
from keras.layers import SeparableConv2D
from keras import layers"""
from keras.regularizers import l2
import keras


@dataclass
class Model_done:
    emot_model = os.path.join("artifacts", "Model.h5")

@dataclass
class ModelCreate:
    def __init__(self, model_create_config: ModelCreation = ModelCreation()):
        try:
            self.model_create_config = model_create_config
            self.model_done = Model_done()
        except Exception as e:
            raise CustomException(e, sys)

    def emotion_model(self):
        logging.info("Model Creation Starting...")

        try:
            input_data = keras.layers.Input(self.model_create_config.input_shape)
            model = keras.layers.Convolution2D(8,(3,3))(input_data)
            model = keras.layers.BatchNormalization()(model)
            model = keras.layers.Activation('relu')(model)
            my_filters = [16,64,256,512]
            
            for filter in my_filters:
                model = keras.layers.Convolution2D(filter,(3,3),padding='same', kernel_regularizer=l2(4e-5))(model)
                model = keras.layers.BatchNormalization()(model)
                model = keras.layers.Activation('relu')(model)
                model = keras.layers.Convolution2D(filter,(3,3),padding='same',kernel_regularizer=l2(4e-5))(model)
                model = keras.layers.BatchNormalization()(model)
                model = keras.layers.MaxPooling2D(pool_size=(2,2), padding='same')(model)
                model = keras.layers.Activation('relu')(model)
            
            model = keras.layers.Convolution2D(self.model_create_config.class_no,(3,3),padding='same', kernel_regularizer=l2(4e-5))(model)
            model = keras.layers.GlobalMaxPooling2D()(model)
            ret_model = keras.models.Model(input_data, keras.layers.Activation('softmax',name='predictions')(model))
            
            return ret_model
        
        except Exception as e:
            raise CustomException(e, sys)
        
    def model_sum(self):
        try:
            model = self.emotion_model()
            model.compile(
                optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy']
            )
            logging.info(model.summary())

            model.save( self.model_done.emot_model)
            logging.info("Model Structure created")

            return(
                self.model_done.emot_model
            )

        except Exception as e:
            logging.info("Error arises in ModelCreation step")
            raise CustomException(e, sys)
            