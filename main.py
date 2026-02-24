import sys
from src.Components.data_ingestion import DataIngestion
from src.Components.data_transformation import DataTransformation
from src.Components.model_trainer import ModelTrainer
from src.Entity.config_entity import ModelTrainerConfig
from src.exception.exception import ExceptionCustom
from src.logging.logger import logging

if __name__=="__main__":
    try:
        logging.info("O Pipeline de Treino foi iniciado")

        ingestion = DataIngestion()
        data_ingestion_artifact = ingestion.init_data_ingestion()
        
        logging.info(f"Ingestão concluída. Ficheiros em: {data_ingestion_artifact.train_file_path}")

        data_transformation = DataTransformation()
        data_transformation_artifact = data_transformation.init_data_transformation(
            data_ingestion_artifact=data_ingestion_artifact
        )
        logging.info("Transformação concluída com sucesso!")

        logging.info("Model Trainer iniciado")
        model_config =ModelTrainerConfig()
        trainer = ModelTrainer(model_config, data_transformation_artifact)
        model_trainer_artifact = trainer.init_model_trainer(
            data_transformation_artifact=data_transformation_artifact
        )
        logging.info("Model Trainer concluído com sucesso!"
)


    except Exception as e:
        raise ExceptionCustom(e,sys)