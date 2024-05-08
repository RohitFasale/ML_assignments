#Assignment No. 02 - Predict Brain weight by body weight
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score


df = pd.read_csv("animals_data.csv")

x = df["Body Weight"]
y = df["Brain Weight"]

X = x.values.reshape(-1,1)

print(X)
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

plt.scatter(X_test, y_test, color = 'grey', label = 'Actual data')
plt.plot(X_test, y_pred, color="blue", linewidth = 2, label='Reg line')
plt.xlabel('Body Weight')
plt.ylabel('Brain Weight')
plt.title("Prediction of brain Weight from body weight")
plt.legend()
plt.show()

accuracy = r2_score(y_test, y_pred)
print("accuracy =", accuracy)


coefficient = model.coef_
print("coefficient =", coefficient)




