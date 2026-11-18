# Iris Predictor

A small Flask web app that predicts the species of an iris flower — **setosa**, **versicolor** or **virginica** — from four flower measurements, using a k-nearest-neighbours (KNN) model trained on the classic Iris dataset.

## Features

- Simple web form for entering sepal/petal measurements
- Scaled KNN pipeline (StandardScaler + KNeighborsClassifier)
- Trained model saved with joblib and loaded on startup
- Standalone training script so the model can be retrained at any time

## Project structure

```
Iris-predictor-/
├── app.py              # Flask app: form page and /predict route
├── train_model.py      # Trains the KNN pipeline and saves model.pkl
├── model.pkl           # Saved trained pipeline (StandardScaler + KNN)
├── requirements.txt    # Python dependencies
├── static/
│   └── style.css       # Styling for the form and result pages
└── templates/
    ├── form.html       # Input form
    └── result.html     # Prediction result page
```

## Requirements

- Python 3.10+ (developed and tested with Python 3.13)
- Dependencies listed in `requirements.txt`: Flask, scikit-learn, joblib, numpy, scipy

## Getting started

```bash
git clone https://github.com/Dsgamers903/Iris-predictor-.git
cd Iris-predictor-

python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
python app.py
```

Then open <http://127.0.0.1:5000> in your browser.

## Usage

Enter the four measurements (in cm) and click **Predict**. The result page shows the predicted species. Some example values:

| Species    | Sepal length | Sepal width | Petal length | Petal width |
|------------|--------------|-------------|--------------|-------------|
| Setosa     | 5.1          | 3.5         | 1.4          | 0.2         |
| Versicolor | 5.9          | 3.2         | 4.8          | 1.8         |
| Virginica  | 6.7          | 3.0         | 5.2          | 2.3         |

## How it works

1. `train_model.py` loads the built-in Iris dataset from scikit-learn (150 samples, 3 classes, 4 features) and splits it 80/20 into training and test sets with `random_state=42`.
2. It builds a pipeline of `StandardScaler` followed by `KNeighborsClassifier(n_neighbors=5)`, trains it, prints the test accuracy (100% on the held-out 30 samples) and saves the pipeline to `model.pkl` with joblib.
3. `app.py` loads the saved pipeline at startup. `GET /` renders the input form, and `POST /predict` feeds the entered values through the pipeline and renders the predicted species name.

## Retraining

Run the training script again to rebuild `model.pkl`:

```bash
python train_model.py
```

The saved model was created with **scikit-learn 1.9.0**. If you upgrade scikit-learn, retrain the model so `model.pkl` stays compatible.

## Notes

- `app.run(debug=True)` starts Flask's development server; use a production WSGI server for real deployments.
- The model is trained on a very small dataset, so predictions for unusual measurements may be unreliable.
