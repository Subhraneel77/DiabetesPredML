import pandas as pd 
import numpy as np 
import sys 
import os
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet 
from sklearn.tree import DecisionTreeRegressor
from src.logger import logging 
from src.exception import CustomException 
from dataclasses import dataclass
from src.utils import save_function 
from src.utils import model_performance 

@dataclass 
class ModelTrainerConfig(): #this will give whatever input I requre with regards to model training.
    trained_model_file_path = os.path.join("artifacts", "model.pkl") # I am dumping my best model With respect to model.pkl


class ModelTrainer():
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, train_array, test_array): 
        try: 
            logging.info("Seggregating the dependent and independent variables")
            X_train, y_train, X_test, y_test = (train_array[:, :-1], #taking out the last column and store everything in train_array
                                                train_array[:,-1], 
                                                test_array[:, :-1], 
                                                test_array[:,-1])
            models = {
                "LinearRegression": LinearRegression(),
                "Ridge": Ridge(), 
                "Lasso":Lasso(), 
                "ElasticNet": ElasticNet(), 
                "DecisionTree": DecisionTreeRegressor()
            }
            model_report: dict = model_performance(X_train, y_train, X_test, y_test, models=models) #this function can be used anywhere as it is written in Utils folder.

            print(model_report)
            print("\n"*100)
            logging.info(f"Model Report: {model_report}")

            # Best model code from dictionary
            best_model_score = max(sorted(model_report.values()))
            # Best model name from dictionary
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)] # Key is the model name and model_report will be getting converted into list and then creating a nested list with respect to the report vales and whatever index is allocated wioth it --> i will get the best model name
            
            best_model = models[best_model_name]

            if best_model_score<0.6: # setting up a threshold
                raise CustomException("No best model found")

            print(f"The best model is {best_model_name}, with R2 Score: {best_model_score}")
            print("\n"*100)
            logging.info(f"The best model is {best_model_name}, with R2 Score: {best_model_score}")
            save_function(file_path= self.model_trainer_config.trained_model_file_path, 
                          obj = best_model)


        except Exception as e: 
            logging.info("Error occured during model training")
            raise CustomException(e,sys)