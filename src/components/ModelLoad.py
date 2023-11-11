from keras.models import model_from_json
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
import sys
from src.utils import ModelLoad

@dataclass
class LoadModel:
    def __init__(self, JsonModel: ModelLoad = ModelLoad):
        self.json_model = ModelLoad

    def model_initiate(self):
        try:
            json_file = open(self.json_model.jsonM, 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            loaded_model = model_from_json(loaded_model_json)

            return loaded_model
        
        except Exception as e:
            raise CustomException(e, sys)