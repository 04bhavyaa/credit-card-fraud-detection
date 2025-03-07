import pandas as pd
import joblib
from src.logger import logging
from src.exception import CustomException
from src.utils import load_object
import sys

# Paths for loading models
MODEL_PATH = "artifacts/rf_model.joblib"
ENCODER_PATH = "artifacts/onehot_encoder.joblib"

class CustomClass:
    def __init__(self, amt, city_pop, job, hour, distance, day_of_week, month, transactions_per_hour,
                 merchant_fraud_rate, user_avg_amt, user_std_amt, yob, category, gender):
        self.amt = amt
        self.city_pop = city_pop
        self.job = job
        self.hour = hour
        self.distance = distance
        self.day_of_week = day_of_week
        self.month = month
        self.transactions_per_hour = transactions_per_hour
        self.merchant_fraud_rate = merchant_fraud_rate
        self.user_avg_amt = user_avg_amt
        self.user_std_amt = user_std_amt
        self.yob = yob
        self.category = category
        self.gender = gender

    def get_data_as_dataframe(self):
        data_dict = {
            "amt": [self.amt],
            "city_pop": [self.city_pop],
            "job": [self.job],
            "hour": [self.hour],
            "distance": [self.distance],
            "day_of_week": [self.day_of_week],
            "month": [self.month],
            "transactions_per_hour": [self.transactions_per_hour],
            "merchant_fraud_rate": [self.merchant_fraud_rate],
            "user_avg_amt": [self.user_avg_amt],
            "user_std_amt": [self.user_std_amt],
            "yob": [self.yob],
            "category": [self.category],
            "gender": [self.gender],
        }
        return pd.DataFrame(data_dict)
    
class PredictionPipeline:
    def __init__(self):
        try:
            logging.info("Loading trained model and encoder...")
            self.model = load_object(MODEL_PATH)
            self.encoder = load_object(ENCODER_PATH)
        except Exception as e:
            raise CustomException(e)

    def preprocess_input(self, data: pd.DataFrame):
        try:
            logging.info("Preprocessing input data...")
            
            # Identify categorical features
            onehot_encode_cols = ['category', 'gender']
            freq_encode_cols = ['job']  # Assuming high-cardinality
            
            # Apply saved encoder - maintain column names
            encoded_array = self.encoder.transform(data[onehot_encode_cols])
            encoded_columns = self.encoder.get_feature_names_out(onehot_encode_cols)
            X_encoded = pd.DataFrame(encoded_array, columns=encoded_columns, index=data.index)
            
            for col in freq_encode_cols:
                data[col] = data[col].map(data[col].value_counts(normalize=True))
            
            # Keep non-categorical features
            data = data.drop(columns=onehot_encode_cols)
            data = pd.concat([data, X_encoded], axis=1)
            
            logging.info("Input data preprocessing complete.")
            return data
        except Exception as e:
            raise CustomException(e, sys)

    def predict(self, data: pd.DataFrame):
        try:
            processed_data = self.preprocess_input(data)
            prediction = self.model.predict(processed_data)
            return prediction
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        "amt": [100.0],
        "city_pop": [50000],
        "job": ["Engineer"],
        "hour": [12],
        "distance": [5.2],
        "day_of_week": [3],
        "month": [5],
        "transactions_per_hour": [10],
        "merchant_fraud_rate": [0.01],
        "user_avg_amt": [50.0],
        "user_std_amt": [20.0],
        "yob": [1985],
        "category": ["grocery_pos"],
        "gender": ["M"]
    })

    pipeline = PredictionPipeline()
    prediction = pipeline.predict(sample_data)
    print("Prediction:", "Fraud" if prediction[0] == 1 else "Legitimate")