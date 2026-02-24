import logging
import os
import pickle
import sys
import pandas as pd
import numpy as np
from sklearn.calibration import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.preprocessing import OneHotEncoder

from src.exception.exception import ExceptionCustom

def read_csv(data_path):
    df = pd.read_csv(data_path)
    return df

def get_transformer_object(train_df, test_df, target_column, preprocessor,cols_to_drop):
    target_feature_train_df = train_df[target_column]
    target_feature_test_df = test_df[target_column]

    drop_columns =cols_to_drop
    
    input_feature_train_df = train_df.drop(columns=drop_columns, axis=1, errors='ignore')
    input_feature_test_df = test_df.drop(columns=drop_columns, axis=1, errors='ignore')
    
    input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
    input_feature_test_arr = preprocessor.transform(input_feature_test_df)

    le = LabelEncoder()
    target_feature_train_arr = le.fit_transform(target_feature_train_df)
    target_feature_test_arr = le.transform(target_feature_test_df)

    if hasattr(input_feature_train_arr, "toarray"):
        input_feature_train_arr = input_feature_train_arr.toarray()
            
    if hasattr(input_feature_test_arr, "toarray"):
        input_feature_test_arr = input_feature_test_arr.toarray()

    train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_arr)]
    test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_arr)]

    return train_arr, test_arr

def get_transformed_object(self):
        try:
            logging.info("Iniciando a criação do objeto preprocessor")
            
            num_features = ['white_rating', 'black_rating']
            cat_features = ['opening_name']

            preprocessor = ColumnTransformer(
                [
                    ("OneHotEncoder", OneHotEncoder(handle_unknown='ignore'), cat_features),
                    ("StandardScaler", StandardScaler(), num_features)
                ]
            )

            return preprocessor

        except Exception as e:
            raise ExceptionCustom(e, sys)


def save_object(file_path, obj):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as file_obj:
        pickle.dump(obj, file_obj)

def save_numpy_array_data(file_path: str, array: np.array):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as file_obj:
        np.save(file_obj, array)

    