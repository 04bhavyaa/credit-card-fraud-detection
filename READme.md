# Credit Card Fraud Detection System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Framework](https://img.shields.io/badge/Framework-Flask-red)
![ML](https://img.shields.io/badge/Library-Scikit--learn-green)

A machine learning system to detect fraudulent credit card transactions in real-time using transaction patterns and customer behavior analysis.

## Features

- Real-time fraud prediction API
- Web interface for transaction verification
- Machine learning pipeline with feature engineering
- Handle class imbalance using careful sampling
- Model interpretability features
- Comprehensive logging and error handling

## Technologies Used

- **Backend**: Python, Flask
- **Machine Learning**: Scikit-learn, Random Forest
- **Data Processing**: Pandas, NumPy
- **Feature Engineering**: One-Hot Encoding, Frequency Encoding
- **Geospatial Analysis**: Geopy
- **Frontend**: HTML5, CSS3

## Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```
Install dependencies
```bash
pip install -r requirements.txt
```

## Usage
Start the web application
```bash
python app.py
```
Visit http://localhost:5000 in your browser

## Make predictions via API
```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
        "amt": 150.0,
        "city_pop": 500000,
        "job": "Engineer",
        "hour": 14,
        "distance": 12.5,
        "day_of_week": 3,
        "month": 7,
        "transactions_per_hour": 15,
        "merchant_fraud_rate": 0.02,
        "user_avg_amt": 75.0,
        "user_std_amt": 25.0,
        "yob": 1985,
        "category": "grocery_pos",
        "gender": "M"
      }'
```
## Project Structure
```
credit-card-fraud-detection/
├── dataset/               # Training and test data
├── artifacts/             # Saved models and encoders
├── notebooks/             # Jupyter notebooks for EDA
├── src/                   # Core application logic
│   ├── pipeline/          # Training and prediction pipelines
│   ├── logger.py          # Logging configuration
│   ├── exception.py       # Custom exception handling
│   └── utils.py           # Utility functions
├── templates/             # HTML templates
├── static/                # CSS and static files
├── app.py                 # Flask application
├── main.py                # CLI interface
├── requirements.txt       # Dependencies
└── setup.py               # Package configuration
```
## Key Features
- Data Preprocessing
- Temporal feature extraction (hour, day, month)
- Geospatial distance calculation
- Frequency encoding for high-cardinality features
- One-hot encoding for categorical variables
- Transaction pattern analysis

## Model Training
- Random Forest Classifier with class weighting
- Automated pipeline for retraining
- Model persistence with joblib
- Comprehensive evaluation metrics

## Fraud Detection Features
- Transaction amount analysis
- Location mismatch detection
- Unusual time activity detection
- Merchant risk profiling
- User spending pattern analysis

## Results
Model Performance
| Metric | Training Score | Validation Score | |---------------|----------------|------------------| | Accuracy | 99.86% | 99.88% | | Precision (1) | 98% | 92% | | Recall (1) | 77% | 75% | | F1-Score (1) | 86% | 83% |

## Web Interface
Fraud Detection Interface

## Limitations & Future Work
- Current placeholder values for some features
- Limited to historical transaction patterns

## Potential improvements:
- Real-time feature calculation
- Deep learning approaches
- Graph-based fraud detection
- Ensemble methods

## Contact
Bhavya Jha - bhavyajha1404@gmail.com

Project Link: https://github.com/04bhavyaa/credit-card-fraud-detection



Bookmark message
Copy message


