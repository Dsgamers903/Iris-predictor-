from flask import Flask, render_template, request
import joblib

model = joblib.load('model.pkl')
app = Flask(__name__)


@app.route('/')
def form():
    return render_template('form.html')


@app.route('/predict', methods=['POST'])
def predict():

    sepal_length = float(request.form['sepal_length'])
    sepal_width = float(request.form['sepal_width'])
    petal_length = float(request.form['petal_length'])
    petal_width = float(request.form['petal_width'])

    features = [[sepal_length, sepal_width, petal_length, petal_width]]

    model_prediction = model.predict(features)

    species = ['setosa', 'versicolor', 'virginica']

    return render_template('result.html', prediction=species[model_prediction[0]])


app.run(debug=True)
