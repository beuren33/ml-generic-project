import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass
from src.exception.exception import ExceptionCustom
from src.logging.logger import logging
from sklearn.model_selection import train_test_split
from src.Entity.config_entity import DataIngestionConfig


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    def init_data_ingestion(self):
        logging.info("Entrando na Ingestão de Dados")
        try:
            df=pd.read_csv('data\games.csv')
            logging.info("Read Data")

            os.makedirs(self.ingestion_config.data_ingestion_dir,exist_ok=True)

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info("Train Test Split")

            train_set,test_set=train_test_split(df,test_size=0.3,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Data Ingestion Completo")


            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise ExceptionCustom(e,sys)
        
if __name__=="__main__":
    obj=DataIngestion()
    obj.init_data_ingestion()

