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
            ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

            model_path = os.path.join(ROOT, "artifacts/model_trainer/model.pkl")
            preprocessor_path = os.path.join(ROOT, "artifacts/data_transformation/preprocessor/preprocessor.pkl")

            model = load_object(model_path)
            preprocessor = load_object(preprocessor_path)

            transformed = preprocessor.transform(features)
            return model.predict_proba(transformed)

        except Exception as e:
            raise ExceptionCustom(e,sys)
        
class CustomData:
    def __init__(self,white_rating:int,black_rating:int,opening_name:str):
        self.white_rating=white_rating
        self.black_rating=black_rating
        self.opening_name=opening_name

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