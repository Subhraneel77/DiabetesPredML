import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer

# from utils import fetch_data_from_mysql # fetching the dataset from MySQL workbench

## Step1: Create path variables to store the files are raw csv
@dataclass #The @dataclass decorator is used to automatically generate base functionalities to classes, including __init__() , __hash__() , __repr__() and more, which helps reduce some boilerplate code.
class DataIngestionconfig:
    train_data_path: str=os.path.join('artifacts','train.csv') #all the outputs of train.csv(file name) will basically gets stored into the artifacts folder.
    test_data_path: str=os.path.join('artifacts','test.csv')
    raw_data_path: str=os.path.join('artifacts','raw.csv')

## create a class for Data Ingestion
class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionconfig()

    def initiate_data_ingestion(self): #
        logging.info('Data Ingestion methods Starts')
        try:
            # Fetching data
            #df = fetch_data_from_mysql()

            #df=pd.read_csv('notebooks/data/student.csv')
            df=pd.read_csv(os.path.join('notebooks/data','diabetespred.csv')) # here I can read the database from MongoDB, Mysql etc.
            logging.info('Dataset read as pandas Dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True) # getting the directionary name with respect to this file path
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True) 
            logging.info('Train test split')
            train_set,test_set=train_test_split(df,test_size=0.30,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True) #done the splitting and saving the train file inside the artifact folder
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info('Ingestion of Data is completed')

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
  
            
        except Exception as e:
            logging.info('Exception occured at Data Ingestion stage')
            raise CustomException(e,sys)

if __name__=="__main__":
    obj=DataIngestion()
    #obj.initiate_data_ingestion()
    train_data,test_data=obj.initiate_data_ingestion()

    data_transformation=DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)

    modeltrainer=ModelTrainer()
    print(modeltrainer.initiate_model_training(train_arr,test_arr)) #this is going me my r2_score
        