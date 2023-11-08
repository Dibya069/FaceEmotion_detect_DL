import os

INPUT_SHAPE = (64, 64, 1)

"""
Data Ingestion related constant
"""
IMG_SIZE: tuple = INPUT_SHAPE[:2]

GET_DATA: str = "C:/Users/mohan/Downloads/Data_Science/008.Deep_Learning/00. Projects/FaceEmotion/FaceEmotionDetect/dataset/fer2013.csv" 


"""
Model Creation related constant 
"""
CLASS_NO: int = 7
BATCH_NO: int = 32


"""
Face Detection related constant
"""
DEMO_IMG: str = "C:/Users/mohan/Downloads/Data_Science/008.Deep_Learning/00. Projects/FaceEmotion/FaceEmotionDetect/tom.jpg"
DEMO_PATH: str = "C:/Users/mohan/Downloads/Data_Science/008.Deep_Learning/00. Projects/FaceEmotion/FaceEmotionDetect/dataset"