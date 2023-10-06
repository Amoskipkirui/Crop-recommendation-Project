from flask import Flask, render_template, request, jsonify
import pickle

# Load the pickled model
with open('best_rf_classifier.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input features from the form
        N = float(request.form['N'])
        K = float(request.form['K'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        pH = float(request.form['pH'])
        rainfall = float(request.form['rainfall'])
        
        # Prepare the features for prediction
        features = [[N, K, temperature, humidity, pH, rainfall]]

        # Perform prediction using the loaded model
        prediction = model.predict(features)

        # Return the prediction to the result template
        return render_template('result.html', prediction=prediction[0])
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
