from flask import Flask, request, render_template
import joblib
import pandas as pd

# Load the saved Random Forest model
model = joblib.load('diabetes_model.pkl')

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get user input from the form
    gender = int(request.form['gender'])
    age = float(request.form['age'])
    hypertension = int(request.form['hypertension'])
    heart_disease = int(request.form['heart_disease'])
    smoking_history = int(request.form['smoking_history'])
    bmi = float(request.form['bmi'])
    hba1c = float(request.form['hba1c'])
    glucose = float(request.form['glucose'])

    # Create a DataFrame for prediction
    input_data = pd.DataFrame({
        'gender': [gender],
        'age': [age],
        'hypertension': [hypertension],
        'heart_disease': [heart_disease],
        'smoking_history': [smoking_history],
        'bmi': [bmi],
        'HbA1c_level': [hba1c],
        'blood_glucose_level': [glucose]
    })

    # Make prediction
    prediction = model.predict(input_data)[0]
    result = "Positive for Diabetes" if prediction == 1 else "Negative for Diabetes"

    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
