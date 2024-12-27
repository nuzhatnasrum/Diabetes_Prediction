import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the dataset
data = pd.read_csv('diabetes.csv')

# Preprocess data
data['gender'] = data['gender'].map({'Male': 0, 'Female': 1})
data['smoking_history'] = data['smoking_history'].map({'never': 0, 'former': 1, 'current': 2, 'No Info': 3})

X = data.drop('diabetes', axis=1)
y = data['diabetes']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, 'diabetes_model.pkl')
