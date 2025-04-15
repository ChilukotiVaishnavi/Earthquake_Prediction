from flask import Flask, request, render_template
import numpy as np
import joblib

app = Flask(__name__)

# Load all models
model_rf = joblib.load(open('random_forest_model.pkl', 'rb'))
model_ada = joblib.load(open('adaboost_model.pkl', 'rb'))
model_svm = joblib.load(open('svm_model.pkl', 'rb'))
model_xgb = joblib.load(open('xgboost_model.pkl', 'rb'))
model_lin = joblib.load(open('linear_regression_model.pkl', 'rb'))
model_xgb_reg = joblib.load(open('xgboost_reg_model.pkl', 'rb'))  # Most accurate assumed

@app.route('/')
def home():
    return render_template('homepage.html')

@app.route('/prediction', methods=['POST'])
def prediction():
    try:
        # Collect and convert input
        data1 = float(request.form['a'])  # Latitude
        data2 = float(request.form['b'])  # Longitude
        data3 = float(request.form['c'])  # Height

        arr = np.array([[data1, data2, data3]])

        # Predict using the most accurate model (XGBoost Regressor)
        pred = model_xgb_reg.predict(arr).item()
        pred_rounded = round(pred, 2)

        # Risk level calculation
        def get_risk(val):
            if val < 4:
                return 'No'
            elif val < 6:
                return 'Low'
            elif val < 8:
                return 'Moderate'
            elif val < 9:
                return 'High'
            else:
                return 'Very High'

        risk = get_risk(pred)

        # Return output using the template
        return render_template('prediction.html', p=pred_rounded, q=risk)

    except Exception as e:
        # Handle any errors and still render template
        return render_template('prediction.html', p="Invalid Input", q="Please enter even floats.")

if __name__ == "__main__":
    app.run(debug=True)
