import os 
import sys 

import pickle 
import numpy as np 
import pandas as pd
import dill
from src.exception import CustomException
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from src.logger import logging

import pymysql # used for fetching data from MySQL workbench



def save_function(file_path, obj): #it will have my file_path and obj then iis going to make the directory according to the particular file path and is going to dump it.
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok= True)
        with open (file_path, "wb") as file_obj: 
            pickle.dump(obj, file_obj) # when we dump the obj then the object will be saved in this specific file path.

    except Exception as e: 
        raise CustomException(e,sys)

def model_performance(X_train, y_train, X_test, y_test, models,param): 
    try: 
        report = {}
        for i in range(len(models)): 
            model = list(models.values())[i]
            para=param[list(models.keys())[i]]

            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)
        
# Train models
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)
# Test data
            y_train_pred =model.predict(X_train)
            y_test_pred = model.predict(X_test)
            #R2 Score 
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)
            report[list(models.keys())[i]] = test_model_score
        return report

    except Exception as e: 
        raise CustomException(e,sys)


def load_obj(file_path): #opening the file in read byte mode nad is loading teh pickel file
    try: 
        with open(file_path, "rb") as file_obj: 
            return pickle.load(file_obj)
    except Exception as e: 
        logging.info("Error in load_object fuction in utils")
        raise CustomException(e,sys)

def fetch_data_from_mysql():
    # Connect to MySQL database
    connection = pymysql.connect(host='root',
                                 user='liveconnection',
                                 password='*******',
                                 database='student')
    
    # Fetch data from MySQL database
    query = "SELECT * FROM studentmarksproject"
    df = pd.read_sql(query, connection)
    
    # Close the connection
    connection.close()
    
    return df