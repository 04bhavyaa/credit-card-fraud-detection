import argparse
import pandas as pd
from src.pipeline.train_pipeline import TrainPipeline
from src.pipeline.predict_pipeline import PredictionPipeline, CustomClass

def train_model(data_path="dataset/cleaned_train_data.csv"):
    """
    Trains the fraud detection model.
    """
    trainer = TrainPipeline(data_path=data_path)
    trainer.run_pipeline()
    print("✅ Model training completed and saved successfully.")

def predict_fraud(input_data):
    """
    Predicts fraud based on input transaction details.
    """
    user_data = CustomClass(**input_data)
    input_df = user_data.get_data_as_dataframe()
    
    pipeline = PredictionPipeline()
    prediction = pipeline.predict(input_df)[0]

    result = "❌ Fraudulent Transaction" if prediction == 1 else "✅ Legitimate Transaction"
    print(f"Prediction: {result}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fraud Detection CLI")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--data-path", type=str, help="Path to training data")
    parser.add_argument("--predict", nargs="+", help="Predict fraud based on input values")

    args = parser.parse_args()

    if args.train:
        data_path = args.data_path if args.data_path else "dataset/cleaned_train_data.csv"
        train_model(data_path)

    elif args.predict:
        # Example: --predict amt=50 city_pop=100000 job=Engineer hour=12 ...
        input_dict = {}
        for item in args.predict:
            key, value = item.split("=")
            if key in ["amt", "distance", "user_avg_amt", "user_std_amt", "merchant_fraud_rate"]:
                input_dict[key] = float(value)
            elif key in ["city_pop", "hour", "day_of_week", "month", "transactions_per_hour", "yob", "gender_M"]:
                input_dict[key] = int(value)
            else:
                input_dict[key] = value
        
        predict_fraud(input_dict)