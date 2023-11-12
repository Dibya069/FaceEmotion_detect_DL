import pandas as pd
import numpy as np
import os, sys
import matplotlib.pyplot as plt
import cv2

#from src.exception import CustomException
from src.utils import FaceDetec, ModelLoad
from src.logger import logging
from src.components.data_ingestion import dataset_provider
from src.components.ModelCreation import ModelCreate
from src.components.FaceDetect import FacetDetect
from src.components.ModelLoad import LoadModel

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
    

    ## Model Load
    obj3 = LoadModel()
    loaded_model = obj3.model_initiate()
    loaded_model.load_weights(ModelLoad.orgM)  # load weights into new model
    
    """
    ## Face Detection for Image
    obj2 = FacetDetect()
    xs, ys, face, cropped = obj2.detect_face(FaceDetec.demo_img)

    j=0
    for i in cropped:
        emotion_pred = int(np.argmax(loaded_model.predict(i)))
        cv2.putText (face, ModelLoad.emolib[emotion_pred], (xs[j], ys[j]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        j+=1
    
    plt.imshow(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))

    cv2.imwrite(FaceDetec.demo_path, face)
    print(FaceDetec.demo_path)
    logging.info("Face Detection step complete")
    """

    ## Face Detection for Video
    obj4 = FacetDetect()
    obj4.web_cam(1)