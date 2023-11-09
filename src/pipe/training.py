import pandas as pd
import numpy as np
import os, sys
import matplotlib.pyplot as plt
import cv2

#from src.exception import CustomException
from src.utils import FaceDetec
from src.logger import logging
from src.components.data_ingestion import dataset_provider
from src.components.ModelCreation import ModelCreate
from src.components.FaceDetect import FacetDetect

if __name__ == "__main__":
    ## Data Ingestion
    obj = dataset_provider()
    train_face, test_face, train_emotion, test_emotion = obj.dataset_initializer()
    print(train_face, test_face, train_emotion, test_emotion)
    logging.info("Data Ingestion step complete")

    ## Model Creation
    obj1 = ModelCreate()
    model_str = obj1.model_sum()
    print(model_str)
    logging.info("Model Creation step complete")

    ## Face Detection
    obj2 = FacetDetect()
    xs, ys, face, cropped = obj2.detect_face(FaceDetec.demo_img)
    
    plt.imshow(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
    plt.show()

    cv2.imwrite(FaceDetec.demo_path, face)
    print(FaceDetec.demo_path)
    logging.info("Face Detection step complete")