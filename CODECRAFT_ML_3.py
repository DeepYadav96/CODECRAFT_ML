
import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


import pandas as pd
df = pd.read_csv('cat_dog_images.csv')
labels = df['label'].values
data = df.drop('label', axis=1).values


X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)


svm = SVC(kernel='linear')
svm.fit(X_train, y_train)


y_pred = svm.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {acc*100:.2f}%")


