# Task-2 (Credit Card Fraud Detection)

Goal: Our goal is to build a machine learning model that can classify transactions as fraudulent (is_fraud = 1) or legitimate (is_fraud = 0).
Technologies: Classification Algorithms (Logistic Regression, Decision Trees, Random Forests), Fraud Detection Techniques, Feature Engineering.


## Here's a breakdown of the features:
### Transaction details:
- trans_date_trans_time: Timestamp of the transaction.
- cc_num: Credit card number (masked or partially anonymized).
- merchant: Merchant where the transaction occurred.
- category: Category of the merchant.
- amt: Transaction amount.
- trans_num: Unique transaction identifier.
- unix_time: Unix timestamp of the transaction.

### Customer details:
- first, last: First and last names.
- gender: Gender of the cardholder.
- dob: Date of birth.
- job: Occupation.

### Location details:
- street, city, state, zip: Address details.
- lat, long: Latitude and longitude of the cardholder’s location.
- city_pop: Population of the city.
- merch_lat, merch_long: Latitude and longitude of the merchant.

### Fraud label:
- is_fraud: Indicates whether the transaction was fraudulent (1) or not (0).