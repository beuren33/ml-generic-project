import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass
from src.exception.exception import ExceptionCustom
from src.logging.logger import logging
from src.Entity.config_entity import DataTransformationConfig
from src.Entity.artifacts_config import DataIngestionArtifact, DataTransformationArtifact
from src.utils.main_utils import get_transformed_object, read_csv, save_numpy_array_data, save_object, get_transformer_object
from sklearn.calibration import LabelEncoder
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    def init_data_transformation(self,data_ingestion_artifact:DataIngestionArtifact):
        try:
            logging.info("Entrando na Transformação de Dados")

            os.makedirs(self.data_transformation_config.data_transformation_dir,exist_ok=True)
            os.makedirs(self.data_transformation_config.preprocessed_object_dir,exist_ok=True)

            logging.info("Diretorios criados!")

            logging.info("Read Data")
            df_train=read_csv(data_ingestion_artifact.train_file_path)
            df_test=read_csv(data_ingestion_artifact.test_file_path)

            logging.info("drop the columns")
            cols_to_drop = ['winner', 'id', 'white_id', 'black_id', 'moves', 'last_move_at', 'victory_status', 'turns']

            preprocessor_obj = get_transformed_object(self)
            target_column = "winner"

            train_arr, test_arr = get_transformer_object(
                train_df=df_train,
                test_df=df_test,
                target_column=target_column,
                preprocessor=preprocessor_obj,
                cols_to_drop=cols_to_drop
            )
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, test_arr)

            save_object(
                file_path=self.data_transformation_config.preprocessed_object_file_path,
                obj=preprocessor_obj
            )

            data_transformation_artifact= DataTransformationArtifact(
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
                preprocessed_object_file_path=self.data_transformation_config.preprocessed_object_file_path
            )
            return data_transformation_artifact
        except Exception as e:
            raise ExceptionCustom(e,sys)



