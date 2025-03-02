import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object

# Paths for saving models
MODEL_PATH = "artifacts/rf_model.joblib"
ENCODER_PATH = "artifacts/onehot_encoder.joblib"

class TrainPipeline:
    def __init__(self, data_path: str):
        self.data_path = data_path

    def load_data(self):
        """Load dataset from the provided path."""
        try:
            logging.info(f"Loading data from {self.data_path}")
            df = pd.read_csv(self.data_path)
            return df
        except Exception as e:
            raise CustomException(e)

    def preprocess_data(self, df):
        """Handle categorical variables using OneHotEncoding."""
        try:
            logging.info("Preprocessing data...")
            
            # Define features and target
            X = df.drop(columns=["is_fraud"])
            y = df["is_fraud"]
            
            # Identify categorical features
            categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
            
            # Apply OneHotEncoding
            encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            X_encoded = pd.DataFrame(encoder.fit_transform(X[categorical_features]))
            
            # Keep non-categorical features
            X = X.drop(columns=categorical_features)
            X = pd.concat([X, X_encoded], axis=1)
            
            logging.info("Data preprocessing complete.")
            return X, y, encoder
        except Exception as e:
            raise CustomException(e)

    def train_model(self, X_train, y_train):
        """Train the RandomForest model."""
        try:
            logging.info("Training RandomForest model...")
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            logging.info("Model training complete.")
            return model
        except Exception as e:
            raise CustomException(e)

    def evaluate_model(self, model, X_test, y_test):
        """Evaluate model performance."""
        try:
            logging.info("Evaluating model...")
            y_pred = model.predict(X_test)
            report = classification_report(y_test, y_pred)
            logging.info(f"Model Evaluation:\n{report}")
        except Exception as e:
            raise CustomException(e)

    def run_pipeline(self):
        """Execute the full training pipeline."""
        try:
            df = self.load_data()
            X, y, encoder = self.preprocess_data(df)
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train model
            model = self.train_model(X_train, y_train)
            
            # Evaluate model
            self.evaluate_model(model, X_test, y_test)
            
            # Save model and encoder
            save_object(MODEL_PATH, model)
            save_object(ENCODER_PATH, encoder)

            logging.info(f"Model saved at {MODEL_PATH}")
            logging.info(f"Encoder saved at {ENCODER_PATH}")
        except Exception as e:
            raise CustomException(e)

if __name__ == "__main__":
    pipeline = TrainPipeline(data_path="dataset/cleaned_train_data.csv")
    pipeline.run_pipeline()
