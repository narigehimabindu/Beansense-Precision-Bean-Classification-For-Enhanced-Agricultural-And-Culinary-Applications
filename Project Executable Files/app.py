import numpy as np
import pickle
import pandas as pd
import os
from flask import Flask, request, render_template

app = Flask(__name__)

# Ensure the model file exists in the specified path
model_path = 'model.pkl'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"No such file or directory: '{model_path}'")

# Load the model
with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/')
def homepage():
    return render_template('homepage.html')

@app.route('/aboutpage')
def aboutpage():
    return render_template('aboutpage.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            input_features = [float(x) for x in request.form.values()]
            if len(input_features) != 16:
                raise ValueError("Expected 16 input features, got {}".format(len(input_features)))
            
            x = [np.array(input_features)]
            
            names = [
                'Area', 'Perimeter', 'MajorAxisLength', 'MinorAxisLength', 'AspectRation', 'Eccentricity',
                'ConvexArea', 'EquivDiameter', 'Extent', 'Solidity', 'roundness', 'Compactness',
                'ShapeFactor1', 'ShapeFactor2', 'ShapeFactor3', 'ShapeFactor4'
            ]
            
            data = pd.DataFrame(x, columns=names)
            
            prediction = model.predict(data)
            
            # Ensure the prediction is an integer
            if isinstance(prediction[0], np.integer):
                prediction_index = int(prediction[0])
            elif isinstance(prediction[0], str):
                # Convert string to index based on prediction labels
                prediction_labels = ['SEKER', 'BARBUNYA', 'BOMBAY', 'CALI', 'HOROZ', 'SIRA', 'DERMASON']
                prediction_index = prediction_labels.index(prediction[0])
            else:
                raise ValueError("Unexpected prediction type: {}".format(type(prediction[0])))
            
            result = prediction_labels[prediction_index]
            
            return render_template("result.html", prediction=result)
        except Exception as e:
            return str(e), 400
    return render_template('predict.html')

if __name__ == "__main__":
    app.run(debug=True, port=5000)
