
# Food Image Recognition and Calorie Estimation Model (using simulated image data from CSV)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error

# Load food image data from CSV (each row: label, calorie, pixel1, pixel2, ...)
# Example CSV: food_images.csv
# label,calorie,pixel1,pixel2,...
df = pd.read_csv('food_images.csv')
labels = df['label'].values  # food item class
calories = df['calorie'].values  # calorie content
data = df.drop(['label', 'calorie'], axis=1).values

# Split data for classification (food recognition)
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Food Recognition Accuracy:", accuracy_score(y_test, y_pred))

# Split data for regression (calorie estimation)
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(data, calories, test_size=0.2, random_state=42)
reg = RandomForestRegressor(n_estimators=100, random_state=42)
reg.fit(X_train_reg, y_train_reg)
y_pred_reg = reg.predict(X_test_reg)
print("Calorie Estimation MAE:", mean_absolute_error(y_test_reg, y_pred_reg))

# Take user input for prediction (pixel values)
try:
    user_pixels = input("Enter 10 pixel values separated by commas (e.g. 12,34,56,...): ")
    user_pixels = [int(x.strip()) for x in user_pixels.split(",")]
    if len(user_pixels) != data.shape[1]:
        raise ValueError("Incorrect number of pixel values.")
    user_img = np.array(user_pixels).reshape(1, -1)
    pred_label = clf.predict(user_img)
    pred_calorie = reg.predict(user_img)
    print('Predicted food label:', pred_label[0])
    print('Estimated calories:', pred_calorie[0])
except Exception as e:
    print("Invalid input. Skipping user prediction.")

# How to run:
# 1. Prepare a CSV file named 'food_images.csv' with columns: label,calorie,pixel1,pixel2,...
# 2. Run this script using:
#    C:/Users/hp/AppData/Local/Programs/Python/Python313/python.exe task5.py
