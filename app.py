from flask import Flask, render_template, request
import pickle
import numpy as np

# Initialize the Flask application
app = Flask(__name__)

# Load the pre-trained model (ensure you have the correct model path)
with open('diabetes_model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')  # HTML form for input

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get form data (user input)
        pregnancies = float(request.form['pregnancies'])
        glucose = float(request.form['glucose'])
        bloodpressure = float(request.form['bloodpressure'])
        skinthickness = float(request.form['skinthickness'])
        insulin = float(request.form['insulin'])
        bmi = float(request.form['bmi'])
        dpf = float(request.form['dpf'])
        age = float(request.form['age'])

        # Create an array of the inputs to feed into the model
        input_data = np.array([[pregnancies, glucose, bloodpressure, skinthickness, insulin, bmi, dpf, age]])

        # Predict using the model
        prediction = model.predict(input_data)

        # Return prediction result
        if prediction[0] == 0:
            result = "The person does not have diabetes."
        else:
            result = "The person has diabetes."

        return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
