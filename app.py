from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from src.pipeline.predict_pipeline import PredictionPipeline, CustomClass

app = Flask(__name__)

# Load transaction history CSV
TRANSACTION_DATA_PATH = "data/transactions.csv"
df_transactions = pd.read_csv(TRANSACTION_DATA_PATH)

def compute_derived_features(user_id, merchant_id):
    """
    Computes derived features for a given user & merchant based on historical transactions stored in CSV.
    """
    user_data = df_transactions[df_transactions["user_id"] == user_id]
    merchant_data = df_transactions[df_transactions["merchant_id"] == merchant_id]

    if user_data.empty:
        user_avg_amt, user_std_amt = 50.0, 20.0  # Default values if no user history
    else:
        user_avg_amt = user_data["amt"].mean()
        user_std_amt = user_data["amt"].std() if len(user_data) > 1 else 1.0  # Avoid division by zero

    if merchant_data.empty:
        merchant_fraud_rate = 0.02  # Default fraud rate if no merchant history
    else:
        merchant_fraud_rate = merchant_data["is_fraud"].mean()  # Fraud cases percentage

    recent_transactions = user_data[user_data["hour"] >= user_data["hour"].max() - 1]
    transactions_per_hour = len(recent_transactions)

    return user_avg_amt, user_std_amt, merchant_fraud_rate, transactions_per_hour


@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        try:
            # User inputs
            user_id = request.form["user_id"]
            merchant_id = request.form["merchant_id"]
            amt = float(request.form["amt"])
            city_pop = int(request.form["city_pop"])
            job = request.form["job"]
            hour = int(request.form["hour"])
            distance = float(request.form["distance"])
            day_of_week = int(request.form["day_of_week"])
            month = int(request.form["month"])
            yob = int(request.form["yob"])
            category = request.form["category"]
            gender_M = int(request.form["gender_M"])

            # Compute derived features
            user_avg_amt, user_std_amt, merchant_fraud_rate, transactions_per_hour = compute_derived_features(user_id, merchant_id)

            # Create CustomClass object
            user_data = CustomClass(
                amt=amt,
                city_pop=city_pop,
                job=job,
                hour=hour,
                distance=distance,
                day_of_week=day_of_week,
                month=month,
                transactions_per_hour=transactions_per_hour,
                merchant_fraud_rate=merchant_fraud_rate,
                user_avg_amt=user_avg_amt,
                user_std_amt=user_std_amt,
                yob=yob,
                category=category,
                gender_M=gender_M
            )

            # Convert input to DataFrame
            input_df = user_data.get_data_as_dataframe()

            # Make prediction
            pipeline = PredictionPipeline()
            prediction = pipeline.predict(input_df)[0]

            # Map result
            result = "❌ Fraudulent Transaction" if prediction == 1 else "✅ Legitimate Transaction"

            return render_template("index.html", result=result)

        except Exception as e:
            return render_template("index.html", result=f"Error: {str(e)}")

    return render_template("index.html", result=None)


if __name__ == "__main__":
    app.run(debug=True)
