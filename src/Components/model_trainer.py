import os
import sys
from dataclasses import dataclass

import numpy as np
from src.Entity.config_entity import ModelTrainerConfig
from src.exception.exception import ExceptionCustom
from src.logging.logger import logging
from src.Entity.artifacts_config import DataTransformationArtifact, ModelTrainerArtifact
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression 
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score,log_loss

from src.utils.main_utils import evaluate_models, save_object

class ModelTrainer:
    def __init__(self,model_trainer_config:ModelTrainerConfig,data_transformation_artifact:DataTransformationArtifact):
        self.model_trainer_config = model_trainer_config
        self.data_transformation_artifact = data_transformation_artifact

    def init_model_trainer(self,data_transformation_artifact:DataTransformationArtifact):
        try:
            logging.info("Entrando na Treinamento do Modelo")
            self.train_array = np.load(self.data_transformation_artifact.transformed_train_file_path)
            self.test_array = np.load(self.data_transformation_artifact.transformed_test_file_path)
            logging.info("Read Data")

            logging.info("Start Splitting")
            X_train, y_train, X_test, y_test = (
                self.train_array[:, :-1],
                self.train_array[:, -1],
                self.test_array[:, :-1],
                self.test_array[:, -1]
            )

            models = {
                "Random Forest": RandomForestClassifier(),
                "Gradient Boost": GradientBoostingClassifier(),
                "Logistic Regression": LogisticRegression(),
                "XGBoost": XGBClassifier(),
                #"MLP": MLPClassifier(),
                "AdaBoost": AdaBoostClassifier()
            }
            params = {
                "Random Forest": {
                    "n_estimators": [50],
                    "max_depth": [10, 20],
                    "min_samples_split": [2, 5]
                },
                "Gradient Boost": {
                    "n_estimators": [100],
                    "learning_rate": [0.1],
                    "max_depth": [3]
                },
                "Logistic Regression": {
                    "C": [1],
                    "solver": ['lbfgs'], 
                    "max_iter": [250]
                },
                "XGBoost": {
                    "n_estimators": [100],
                    "learning_rate": [0.1],
                    "max_depth": [3, 6]
                },
                #"MLP": {
                 #   "hidden_layer_sizes": [(100,), (50, 50)],
                  #  "activation": ['relu'],
                   # "solver": ['adam'],
                    #"max_iter": [200]
                #},
                "AdaBoost": {
                    "n_estimators": [50],
                    "learning_rate": [0.1]
                }
            }

            model_report: dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models,params=params)

            best_model_name = max(model_report, key=lambda x: model_report[x]['accuracy'])
            best_model_score = model_report[best_model_name]['accuracy']

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise Exception("Nenhum Modelo Bom foi Encontrado")
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_obj_file_path=self.model_trainer_config.trained_model_file_path
            )
            return model_trainer_artifact


        except Exception as e:
            raise ExceptionCustom(e,sys)