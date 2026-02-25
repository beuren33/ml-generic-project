import logging
import os
import pickle
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, log_loss
from sklearn.calibration import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder
import yaml

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

def load_object(file_path):
    with open(file_path, "rb") as file_obj:
        return pickle.load(file_obj)


def save_numpy_array_data(file_path: str, array: np.array):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as file_obj:
        np.save(file_obj, array)

def evaluate_models(X_train, y_train, X_test, y_test, models,params):
    try:
        report = {}

        for i in range(len(list(models))):
            model_name = list(models.keys())[i]
            model = list(models.values())[i]
            para = params[model_name]

            logging.info(f"Iniciando Hyperparameter Tuning para: {model_name}")

            gs = GridSearchCV(model, para, cv=3, n_jobs=-1, verbose=1)
            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            y_test_pred = model.predict(X_test)
            test_model_score = accuracy_score(y_test, y_test_pred)

            report[model_name] = {
                "accuracy": float(test_model_score),
                "best_params": gs.best_params_
            }

        report_path = os.path.join("artifacts", "model_report.yaml")
        with open(report_path, "w") as f:
            yaml.dump(report, f, default_flow_style=False)
            
        logging.info(f"Relatório de modelos salvo em: {report_path}")

        return report
    
    except Exception as e:
        raise ExceptionCustom(e, sys)

    