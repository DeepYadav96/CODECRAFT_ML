# Linear Regression Model for House Price Prediction
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# Load data from CSV file
df = pd.read_csv('house_prices.csv')

# Take user input for prediction
try:
    sqft = float(input("Enter square footage: "))
    beds = int(input("Enter number of bedrooms: "))
    baths = int(input("Enter number of bathrooms: "))
    user_house = [[sqft, beds, baths]]
except Exception as e:
    print("Invalid input. Using default values (2000, 3, 2).")
    user_house = [[2000, 3, 2]]


X = df[['square_footage', 'bedrooms', 'bathrooms']]
y = df['price']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Take user input for prediction
try:
    sqft = float(input("Enter square footage: "))
    beds = int(input("Enter number of bedrooms: "))
    baths = int(input("Enter number of bathrooms: "))
    user_house = [[sqft, beds, baths]]
except Exception as e:
    print("Invalid input. Using default values (2000, 3, 2).")
    user_house = [[2000, 3, 2]]
predicted_price = model.predict(user_house)
print("Predicted price for your house:", predicted_price[0])