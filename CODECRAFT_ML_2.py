
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


data = {
    'CustomerID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Annual_Spend': [500, 1500, 700, 1200, 3000, 3500, 200, 800, 2200, 2700],
    'Num_Transactions': [5, 15, 7, 12, 30, 35, 2, 8, 22, 27],
    'Avg_Basket_Size': [50, 100, 70, 90, 120, 130, 40, 80, 110, 125]
}
df = pd.DataFrame(data)


X = df[['Annual_Spend', 'Num_Transactions', 'Avg_Basket_Size']]


kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(X)

print(df[['CustomerID', 'Cluster']])


plt.scatter(df['Annual_Spend'], df['Num_Transactions'], c=df['Cluster'], cmap='viridis')
plt.xlabel('Annual Spend')
plt.ylabel('Number of Transactions')
plt.title('Customer Segments')
plt.show()

# Take user input for new customer and predict cluster
try:
    spend = float(input("Enter annual spend: "))
    trans = int(input("Enter number of transactions: "))
    basket = float(input("Enter average basket size: "))
    user_customer = [[spend, trans, basket]]
    cluster = kmeans.predict(user_customer)
    print(f"Predicted cluster for this customer: {cluster[0]}")
except Exception as e:
    print("Invalid input. Skipping user prediction.")
