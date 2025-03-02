from flask import Flask, render_template, request, jsonify
import pandas as pd
from src.pipeline.predict_pipeline import PredictionPipeline, CustomClass
from src.logger import logging
from src.exception import CustomException
import sys

app = Flask(__name__, template_folder='templates')

@app.route('/', methods=['GET', 'POST'])
def home():
    result = None
    if request.method == 'POST':
        try:
            # Extract form data
            amt = float(request.form['amt'])
            city_pop = int(request.form['city_pop'])
            job = request.form['job']
            hour = int(request.form['hour'])
            distance = float(request.form['distance'])
            day_of_week = int(request.form['day_of_week'])
            month = int(request.form['month'])
            
            # These would normally be calculated from historical data
            # For demo purposes, we're using placeholder values or user inputs
            transactions_per_hour = 10  # placeholder
            merchant_fraud_rate = 0.01  # placeholder
            user_avg_amt = float(request.form.get('user_avg_amt', 50.0))
            user_std_amt = float(request.form.get('user_std_amt', 20.0))
            
            yob = int(request.form['yob'])
            category = request.form['category']
            gender_M = int(request.form['gender_M'])
            
            # Create input data object
            input_data = {
                'amt': amt,
                'city_pop': city_pop,
                'job': job,
                'hour': hour,
                'distance': distance,
                'day_of_week': day_of_week,
                'month': month,
                'transactions_per_hour': transactions_per_hour,
                'merchant_fraud_rate': merchant_fraud_rate,
                'user_avg_amt': user_avg_amt,
                'user_std_amt': user_std_amt,
                'yob': yob,
                'category': category,
                'gender_M': gender_M
            }
            
            # Prepare data for prediction
            user_data = CustomClass(**input_data)
            input_df = user_data.get_data_as_dataframe()
            
            # Make prediction
            pipeline = PredictionPipeline()
            prediction = pipeline.predict(input_df)[0]
            
            # Determine result
            result = "❌ Fraudulent Transaction" if prediction == 1 else "✅ Legitimate Transaction"
            
            logging.info(f"Prediction: {result}")
            
        except Exception as e:
            logging.error(f"Error during prediction: {e}")
            result = f"Error during prediction: {str(e)}"
            raise CustomException(e, sys)
    
    return render_template('index.html', result=result)

@app.route('/api/predict', methods=['POST'])
def predict_api():
    try:
        # Get JSON data
        data = request.json
        
        # Create input data object
        user_data = CustomClass(**data)
        input_df = user_data.get_data_as_dataframe()
        
        # Make prediction
        pipeline = PredictionPipeline()
        prediction = pipeline.predict(input_df)[0]
        
        # Return result
        result = {
            "prediction": int(prediction),
            "status": "Fraudulent" if prediction == 1 else "Legitimate"
        }
        
        return jsonify(result)
    
    except Exception as e:
        logging.error(f"API Error: {e}")
        return jsonify({"error": str(e), "status": "failed"}), 400

if __name__ == "__main__":
    app.run(debug=True)