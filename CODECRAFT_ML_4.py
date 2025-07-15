
# Hand Gesture Recognition Model (using simulated image data from CSV)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load gesture data from CSV (each row: label, pixel1, pixel2, ...)
# Example CSV: gesture_images.csv
# label,pixel1,pixel2,...
df = pd.read_csv('gesture_images.csv')
labels = df['label'].values
data = df.drop('label', axis=1).values

# Split data
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Train a classifier (Random Forest)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predict and evaluate
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Take user input for prediction (pixel values)
try:
    user_pixels = input("Enter 10 pixel values separated by commas (e.g. 12,34,56,...): ")
    user_pixels = [int(x.strip()) for x in user_pixels.split(",")]
    if len(user_pixels) != data.shape[1]:
        raise ValueError("Incorrect number of pixel values.")
    user_gesture = np.array(user_pixels).reshape(1, -1)
    pred = clf.predict(user_gesture)
    print('Predicted gesture label:', pred[0])
except Exception as e:
    print("Invalid input. Skipping user prediction.")

# How to run:
# 1. Prepare a CSV file named 'gesture_images.csv' with gesture data (label,pixel1,pixel2,...)
# 2. Run this script using:
#    C:/Users/hp/AppData/Local/Programs/Python/Python313/python.exe task4.py
