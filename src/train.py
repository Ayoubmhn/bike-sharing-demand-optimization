import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 1. Load Dataset
# Path matches the structure: data/Bike.csv
try:
    df = pd.read_csv("data/Bike.csv")
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: data/Bike.csv not found. Please check your folder structure.")
    exit()

# 2. Preprocessing & Feature Engineering
# CRITICAL: We drop 'casual' and 'registered' because their sum is 'count'.
# Keeping them would be "Data Leakage" (predicting the total using its parts).
# We also drop 'datetime' (string) and 'atemp' (multicollinearity with 'temp').
cols_to_drop = ['count', 'casual', 'registered', 'datetime', 'atemp']
X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
y = df['count']

# 3. Train/Test Split (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Feature Scaling
# Essential for consistent model behavior
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Model: XGBoost with Hyperparameter Tuning
# Using GridSearchCV shows advanced understanding of model selection
xgb_reg = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

param_grid = {
    'n_estimators': [100, 300],
    'max_depth': [3, 6, 9],
    'learning_rate': [0.05, 0.1],
    'subsample': [0.8]
}

print("Running Grid Search (5-fold CV)...")
grid_search = GridSearchCV(estimator=xgb_reg, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train_scaled, y_train)

# 6. Evaluation
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test_scaled)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"\n--- Results ---")
print(f"Best Params: {grid_search.best_params_}")
print(f"Final RMSE: {rmse:.2f}")
print(f"R2 Score: {r2:.4f}")

# 7. Feature Importance Analysis
# This allows you to answer the 'Error Analysis' section in the form
plt.figure(figsize=(10, 6))
plt.barh(X.columns, best_model.feature_importances_)
plt.title("XGBoost - Feature Importance")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig("feature_importance.png")
print("\nPlot saved: feature_importance.png")
