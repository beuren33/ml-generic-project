import sys
import os
import pandas as pd
from src.exception.exception import ExceptionCustom
from src.logging.logger import logging
from src.utils.main_utils import load_object

class PredictionPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path =os.path.join('artifacts/model_trainer', 'model.pkl')
            preprocessor_path =os.path.join('artifacts/data_transformation', 'preprocessor', 'preprocessor.pkl')
            
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            data_scaled = preprocessor.transform(features)
            preds_proba = model.predict_proba(data_scaled)
            return preds_proba

        except Exception as e:
            raise ExceptionCustom(e,sys)
        
class CustomData:
    def __init__(self,white_rating:int,black_rating:int,opening_eco:str):
        self.white_rating=white_rating
        self.black_rating=black_rating
        self.opening_name=opening_eco

    def get_data_frame(self):
        try:
            custom_data_input_dict = {
                "white_rating": [self.white_rating],
                "black_rating": [self.black_rating],
                "opening_name": [self.opening_name]
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise ExceptionCustom(e,sys)