import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_obj


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path= os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            print("Before Loading")
            model=load_obj(file_path=model_path)
            preprocessor=load_obj(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features) #transform the features
            preds=model.predict(data_scaled) #my model will do the prediction
            return preds #it will return the prediction
        
        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(  self,
        gender: str,
        smoking_history: str,
        age: int,
        hypertension: int,
        bmi: int,
        HbA1c_level: int,
        blood_glucose_level: int):

        self.gender = gender

        self.smoking_history = smoking_history

        self.age = age

        self.hypertension = hypertension

        self.bmi = bmi

        self.HbA1c_level = HbA1c_level

        self.blood_glucose_level = blood_glucose_level

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "age": [self.age],
                "hypertensionn": [self.hypertension],
                "bmi": [self.bmi],
                "HbA1c_level": [self.HbA1c_level],
                "smoking_history": [self.smoking_history],
                "blood_glucose_level": [self.blood_glucose_level],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)