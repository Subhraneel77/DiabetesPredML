import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler #OneHotEncoder :the input to this transformer should be an array-like of integers or strings, denoting the values taken on by categorical (discrete) features.


from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_function

@dataclass #The @dataclass decorator is used to automatically generate base functionalities to classes, including __init__() , __hash__() , __repr__() and more, which helps reduce some boilerplate code.
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformation_object(self): # This function si responsible for data trnasformation based on a different types of data.
        try:
            logging.info('Data Transformation initiated')
            # Define which columns should be ordinal-encoded and which should be scaled
            categorical_cols = ['gender', 'smoking_history']
            numerical_cols = ['age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level', 'blood_glucose_level']
            
 
            logging.info('Pipeline Initiated')

            ## Numerical Pipeline
            num_pipeline=Pipeline( # this pipeline is used to run on the training dataset like fit_transform in the training dataset and transform into test dataset.
                steps=[
                ('imputer',SimpleImputer(strategy='median')), # this line of code is written to handle the missing numerical values in the dataset i.e, basically replacing the missing values with the median.
                ('scaler',StandardScaler()) # Doing the standard scaling 

                ]

            )

            # Categorigal Pipeline
            cat_pipeline=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='most_frequent')), # this line of code is written to handle the missing categorical values in the dataset ie, basically replacing the missing values with the mode.
                ("one_hot_encoder",OneHotEncoder()), # we can also use targetguided encoder and in EDA we have seen that there were very less number of categories in every categorical variable.
                ('scaler',StandardScaler()) # Doing the standard scaling 
                ]

            )
            #joining two pipelines
            preprocessor=ColumnTransformer([ 
            ('num_pipeline',num_pipeline,numerical_cols),
            ('cat_pipeline',cat_pipeline,categorical_cols)
            ])
            
            return preprocessor

            logging.info('Pipeline Completed')

        except Exception as e:
            logging.info("Error in Data Trnasformation")
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            # Reading train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read train and test data completed')
            
            logging.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head  : \n{test_df.head().to_string()}')

            logging.info('Obtaining preprocessing object')

            preprocessing_obj = self.get_data_transformation_object() #this preprocessing_obj needs to be converted into a pickel file

            target_column_name = 'diabetes'
            
            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            
            ## Transformating using preprocessor obj
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            logging.info("Applying preprocessing object on training and testing datasets.")
            

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_function(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )
            logging.info('Preprocessor pickle file saved')

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
            
        except Exception as e:
            logging.info("Exception occured in the initiate_datatransformation")

            raise CustomException(e,sys)
        
# if __name__=="__main__":
#     obj=DataTransformation()
#     obj.initiate_data_transformation()
    # obj = DataTransformation()  # Assuming DataTransformation is the class containing the initiate_data_transformation method
    # train_path = "artifacts/train.csv"
    # test_path = "artifacts/test.csv"
    # obj.initiate_data_transformation(train_path, test_path)

