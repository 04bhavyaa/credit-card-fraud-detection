from setuptools import setup, find_packages

setup(
    name="credit-card-fraud-detection",
    version="0.0.1",
    author="Bhavya Jha",
    author_email="bhavyajha1404@gmail.com",
    description="Credit Card Fraud Detection",
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'scikit-learn',
        'flask',
        'matplotlib', 
        'seaborn',
        'category_encoders',
        'joblib',
        'imblearn',
        'geopy'
    ]
)