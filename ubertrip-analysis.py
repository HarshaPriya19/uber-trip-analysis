import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb # type: ignore
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
# Load Data
file_paths = ["uber-raw-data-apr14.csv", "uber-raw-data-may14.csv", 
              "uber-raw-data-jun14.csv", "uber-raw-data-jul14.csv",
              "uber-raw-data-aug14.csv", "uber-raw-data-sep14.csv"]

dataframes = [pd.read_csv(file, nrows=100000) for file in file_paths]  # Load 100K rows per file
data = pd.concat(dataframes, ignore_index=True)

# Convert 'Date/Time' to datetime format
data.rename(columns={'Date/Time': 'Datetime'}, inplace=True)
data['Datetime'] = pd.to_datetime(data['Datetime'])

# Extract features
data['Hour'] = data['Datetime'].dt.hour
data['DayOfWeek'] = data['Datetime'].dt.dayofweek
data['Month'] = data['Datetime'].dt.month

# Assume a proxy target for fare prediction (since dataset lacks fare info)
# Let's create a synthetic fare amount using distance between points
data['Lat_diff'] = data['Lat'].diff().fillna(0)
data['Lon_diff'] = data['Lon'].diff().fillna(0)
data['Fare'] = np.sqrt(data['Lat_diff']**2 + data['Lon_diff']**2) * 50  # Approximate fare

# Visualization: Trips per Hour
plt.figure(figsize=(10,6))
sns.countplot(x=data['Hour'], hue=data['Hour'], palette='coolwarm', legend=False)
plt.title('Trips per Hour')
plt.xlabel('Hour of the Day')
plt.ylabel('Trip Count')
plt.show()

# Visualization: Busiest Days
plt.figure(figsize=(10,6))
sns.countplot(x=data['DayOfWeek'], hue=data['DayOfWeek'], palette='viridis', legend=False)
plt.title('Busiest Days for Uber Rides')
plt.xlabel('Day of the Week (0=Monday, 6=Sunday)')
plt.ylabel('Trip Count')
plt.show()

# Feature Engineering for Trip Prediction
trip_counts = data.groupby(['Hour', 'DayOfWeek', 'Month']).size().reset_index(name='Trips')

X = trip_counts[['Hour', 'DayOfWeek', 'Month']]
y = trip_counts['Trips']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a Random Forest model
rfr = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)
rfr.fit(X_train, y_train)

# Predict and evaluate
y_pred = rfr.predict(X_test)

print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R^2 Score:", r2_score(y_test, y_pred))

# Visualize Predictions
plt.figure(figsize=(10,6))
plt.scatter(y_test, y_pred, alpha=0.3)
plt.xlabel('Actual Trips')
plt.ylabel('Predicted Trips')
plt.title('Actual vs Predicted Trips')
plt.show()

# Fare Prediction
X = data[['Hour', 'DayOfWeek', 'Month', 'Lat', 'Lon']]
y = data['Fare']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost model
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 6],
    'learning_rate': [0.01, 0.1]
}

grid_search = GridSearchCV(xgb_model, param_grid, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

best_xgb = grid_search.best_estimator_
y_pred = best_xgb.predict(X_test)

# Model Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Best XGBoost Parameters: {grid_search.best_params_}")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R^2 Score: {r2:.2f}")

# Plot Actual vs Predicted Fares
plt.figure(figsize=(10, 5))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.5)
plt.xlabel('Actual Fare')
plt.ylabel('Predicted Fare')
plt.title('Uber Fare Prediction: Actual vs Predicted')
plt.show()


