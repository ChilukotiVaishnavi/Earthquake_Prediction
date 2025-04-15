import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, RandomForestRegressor
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from xgboost import XGBClassifier, XGBRegressor
import joblib

# Load dataset
data = pd.read_csv("dataset.csv")

# Ensure there are no missing values
data = data.dropna()

# Convert all features and labels to numeric values
X = data.iloc[:, :-1].apply(pd.to_numeric, errors='coerce').values

# Classification target (discrete classes)
y_class = pd.to_numeric(data.iloc[:, -1], errors='coerce').values.astype(int)

# Regression target (continuous values)
y_reg = pd.to_numeric(data.iloc[:, -1], errors='coerce').values.astype(float)

# Debugging: Print unique values of y_class and y_reg to verify
print(f"Unique values in classification target (y_class): {np.unique(y_class)}")
print(f"Unique values in regression target (y_reg): {np.unique(y_reg)}")

# Re-map classification labels to start from 0
# Create a mapping dictionary for class labels
unique_classes = np.unique(y_class)
class_mapping = {label: idx for idx, label in enumerate(unique_classes)}

# Apply the mapping to the target variable
y_class = np.vectorize(class_mapping.get)(y_class)

# Debugging: Print unique values after remapping
print(f"Unique values in remapped classification target (y_class): {np.unique(y_class)}")

# Ensure there are no NaN values after conversion
X = np.nan_to_num(X)
y_class = np.nan_to_num(y_class)

# Train-test split for classification
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_class, test_size=0.2, random_state=42)

# Train-test split for regression
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y_reg, test_size=0.2, random_state=42)

# Debugging: Check if the split data is correct
print(f"Training data for classification (X_train_c): {X_train_c.shape}, {y_train_c.shape}")
print(f"Training data for regression (X_train_r): {X_train_r.shape}, {y_train_r.shape}")

# Classification models (only use y_class as target)
class_models = {
    'random_forest_model.pkl': RandomForestClassifier(n_estimators=100, random_state=42),
    'adaboost_model.pkl': AdaBoostClassifier(n_estimators=100, random_state=42),
    'svm_model.pkl': SVC(probability=True),
    'xgboost_model.pkl': XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric='mlogloss'
    )
}

# Train and save classification models
for name, model in class_models.items():
    print(f"Training model: {name}")
    model.fit(X_train_c, y_train_c)
    joblib.dump(model, open(name, 'wb'))

# Regression models (only use y_reg as target)
# XGBoost Regression Model
xgboost_reg_model = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1)
xgboost_reg_model.fit(X_train_r, y_train_r)
joblib.dump(xgboost_reg_model, open('xgboost_reg_model.pkl', 'wb'))

# Linear Regression model
lin_reg = LinearRegression()
lin_reg.fit(X_train_r, y_train_r)
joblib.dump(lin_reg, open('linear_regression_model.pkl', 'wb'))

# Random Forest Regressor model
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regressor.fit(X_train_r, y_train_r)
joblib.dump(rf_regressor, open('random_forest_regressor_model.pkl', 'wb'))

print("✅ All models trained and saved successfully.")
