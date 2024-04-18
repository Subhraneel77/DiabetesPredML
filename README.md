# Student Marks Prediction Machine Learning Pipeline

### Step 1: Create a new environment

```
conda create -p venv python==3.8

conda activate venv/
```
### Step 2: Create a .gitignore file

```
create the file by right click and include the venv in it
```

### Step 3: Create a requirements.txt file 
```
pip install -r requirements.txt
```

### Step 4: Create a setup.py file 
```
This is to install the entire project as a package. Additionally, write a function to read the packages from requirements.txt
```

### Step5: Create a folder `src` 
```
Include exception, logger, and utils python files. Make this folder as a package by including __init__.py file. The scr folder will include another folder with name components will be created. Include __init__.py also 
```
#### Step 5.1 Create a folder `components`

```
Include data_ingestion, data_transformation, model trainer, and __init_.py. These components are to be interconnected in future. 
```
#### Step 5.2 Create a folder called `pipeline`
```
Create two python files training_pipeline and prediction_pipeline with __init__.py folder
``` 

### Step 6: Create a folder called `notebooks` 
```
Create a folder called data and include the dataset. Additionally, create a EDA.ipynb file to do the EDA analysis.
```
## Overview

This project aims to predict student marks using machine learning techniques. The goal is to develop a robust pipeline that takes in various features related to students and predicts their marks accurately.

## Table of Contents

1. [Installation](#installation)
2. [Usage](#usage)
3. [Dataset](#dataset)
4. [Preprocessing](#preprocessing)
5. [Feature Engineering](#feature-engineering)
6. [Model Training](#model-training)
7. [Evaluation](#evaluation)
8. [Deployment](#deployment)
9. [Contributing](#contributing)
10. [License](#license)

## Installation

To run this project, you need to have Python 3.11 installed. Clone the repository and install the required dependencies using pip:

```bash
git clone https://github.com/yourusername/student-marks-prediction.git
cd student-marks-prediction
pip install -r requirements.txt
```

## Usage

Once you have installed the dependencies, you can use the provided scripts to run different parts of the pipeline. Here are the main scripts:

train.py: Used to train the machine learning model.
predict.py: Predicts the marks for new student data.
evaluate.py: Evaluates the performance of the trained model.

Example usage:

```bash
python train.py --dataset data/train.csv --model rf
```

## Dataset

The dataset used in this project contains information about students such as age, gender, study time, etc., and their corresponding marks. You can find the dataset in the data directory.

## Preprocessing

Before training the model, the dataset undergoes preprocessing steps such as handling missing values, encoding categorical variables, and scaling numerical features. You can find the preprocessing code in preprocessing.py.

## Feature Engineering

Feature engineering involves creating new features or transforming existing ones to improve model performance. The feature engineering code can be found in feature_engineering.py.

## Model Training

The model_training.ipynb notebook contains code for building and training the machine learning model using the preprocessed data. Various algorithms can be explored and compared for model training.

## Evaluation

The model_training.ipynb notebook displays the trained model's performance metrics such as accuracy, precision, recall, and F1-score.

## Deployment

Once satisfied with the model's performance, you can deploy it using various methods such as Flask API, Docker container, or cloud-based solutions.

## Contributing

Contributions are welcome! For major changes, please open an issue first to discuss what you would like to change.

