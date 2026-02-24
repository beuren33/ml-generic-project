import os
import sys
from dataclasses import dataclass
from src.exception.exception import ExceptionCustom
from src.logging.logger import logging

@dataclass
class DataIngestionConfig:
    data_ingestion_dir: str = os.path.join('artifacts', 'data_ingestion')
    train_data_path: str = os.path.join('artifacts/data_ingestion', 'train.csv')
    test_data_path: str = os.path.join('artifacts/data_ingestion', 'test.csv')
    raw_data_path: str = os.path.join('artifacts/data_ingestion', 'data.csv')

@dataclass
class DataTransformationConfig:
    data_transformation_dir: str = os.path.join('artifacts', 'data_transformation')
    transformed_train_file_path: str = os.path.join('artifacts/data_transformation', 'train.npy')
    transformed_test_file_path: str = os.path.join('artifacts/data_transformation', 'test.npy')
    preprocessed_object_dir:str = os.path.join('artifacts/data_transformation', 'preprocessor')
    preprocessed_object_file_path: str = os.path.join('artifacts/data_transformation/preprocessor', 'preprocessor.pkl')

@dataclass
class ModelTrainerConfig:
    model_trainer_dir: str = os.path.join('artifacts', 'model_trainer')
    trained_model_file_path: str = os.path.join('artifacts/model_trainer', 'model.pkl')
    report_file_path: str = os.path.join('artifacts/model_trainer', 'report.yaml')


