from src.data_processing import DataProcessing
from src.model_training import modelTraining

if __name__ == "__main__":
    processor = DataProcessing("artifacts/raw/data.csv", "artifacts/processed")
    processor.run()

    trainer = modelTraining("artifacts/processed", "artifacts/models")
    trainer.run()
    