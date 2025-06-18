🏠 Bangalore House Price Prediction App
This project builds an end-to-end machine learning application to predict house prices in Bangalore using the Bangalore House Prices dataset. The app uses a Linear Regression model trained on features like area, bedrooms, bathrooms, and location, with a Flask-based web interface for user input and price predictions.
📋 Project Overview

Dataset: Bangalore House Prices from Kaggle
Features: Area (sq ft), bedrooms, bathrooms, location
Model: Linear Regression
Frontend: HTML/CSS with Flask
Deployment: Render (free hosting)

🛠️ Project Structure
house-price-app/
├── app.py                    # Flask application
├── bangalore_house_price_model.pkl  # Trained ML model
├── requirements.txt          # Python dependencies
├── templates/
│   └── index.html            # HTML form for user input
├── static/
│   └── style.css             # CSS for styling the form
└── notebook/
    └── Bangalore_House_Price_Prediction.ipynb  # Jupyter Notebook for model training

🚀 Getting Started
Prerequisites

Python 3.8+
Git
A Kaggle account to download the dataset
Render account for deployment (optional)

Step 1: Clone the Repository
git clone https://github.com/your-username/house-price-app.git
cd house-price-app

Step 2: Install Dependencies
Create a virtual environment and install the required packages:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

Step 3: Download the Dataset
Download the Bangalore House Prices dataset from Kaggle and place Bengaluru_House_Data.csv in the notebook/ folder.
Step 4: Train the Model
Open the notebook/Bangalore_House_Price_Prediction.ipynb Jupyter Notebook and run the cells to:

Load and clean the dataset
Perform feature engineering
Train a Linear Regression model
Save the model as bangalore_house_price_model.pkl

Alternatively, use the provided pre-trained model (bangalore_house_price_model.pkl).
Jupyter Notebook Steps
The notebook includes:

Data Loading: Loads Bengaluru_House_Data.csv using Pandas.
Data Cleaning: Drops irrelevant columns (area_type, society, balcony, availability) and handles missing values.
Feature Engineering: Converts total_sqft to numerical values, extracts bhk from size, and reduces unique location values.
Outlier Removal: Filters out houses with unrealistic square footage per bedroom.
Model Training: Trains a Linear Regression model and evaluates it using R² Score and Mean Squared Error (MSE).
Model Saving: Saves the trained model using joblib.

Step 5: Run the Flask App Locally
python app.py

Open http://127.0.0.1:5000 in your browser. Enter house details (area, bedrooms, bathrooms) to get a price prediction.
Step 6: Deploy to Render

Push your code to a GitHub repository.
Create an account at Render.
Click New Web Service and connect your GitHub repository.
Configure:
Build Command: pip install -r requirements.txt
Start Command: python app.py


Deploy the app. Once deployed, access it via the provided Render URL (e.g., https://house-price-predictor.onrender.com).

📄 Files

app.py: Flask application handling the web interface and predictions.
bangalore_house_price_model.pkl: Pre-trained Linear Regression model.
requirements.txt: List of Python dependencies.
templates/index.html: HTML form for user input.
static/style.css: CSS for styling the web interface.
notebook/Bangalore_House_Price_Prediction.ipynb: Jupyter Notebook for data preprocessing and model training.

📦 Generate requirements.txt
To create requirements.txt, run:
pip freeze > requirements.txt

Example content:
flask==2.0.1
pandas==1.5.3
numpy==1.23.5
scikit-learn==1.2.2
joblib==1.2.0

🔧 Improving the Model
To enhance the model:

Try Advanced Algorithms: Use Random Forest or XGBoost for better accuracy.from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


Feature Engineering: Add polynomial features or interaction terms.
Hyperparameter Tuning: Use GridSearchCV to optimize model parameters.
Cross-Validation: Implement k-fold cross-validation for robust evaluation.

📈 Results

R² Score: Measures the proportion of variance explained by the model.
MSE: Quantifies prediction error in squared units.

Output from the notebook:
R² Score: 0.5303123010113816
MSE: 11479.5025915459

🙌 Contributing
Feel free to fork this repository, submit issues, or create pull requests to improve the model or app.

