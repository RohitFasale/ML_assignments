#Assignment No.1 - Predict weight of Animal by Height
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("height_weight_dataset.csv")

# Extracting features (height) and target variable (weight)
x = df['Height(Inches)']
y = df['Weight(Pounds)']

X = x.values.reshape(-1,1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and fit the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
accuracy = r2_score(y_test, y_pred)
print("Mean Squared Error:", mse)
print("Accuracy :", accuracy)

# Plot the regression line
plt.scatter(X_test, y_test, color='black')
plt.plot(X_test, y_pred, color='blue', linewidth=3)
plt.xlabel('Height (Inches)')
plt.ylabel('Weight (Pounds)')
plt.title('Linear Regression: Height vs Weight')
plt.show()

# Predict on new data
user_data = int(input("Enter your height in inches"))

new_data = pd.DataFrame({'Height(Inches)': [user_data]})
predicted_weight = model.predict(new_data)

print("your weight is:", predicted_weight)

#Import Libraries: The code begins by importing necessary libraries such as pandas for data manipulation, scikit-learn for machine learning functionalities, matplotlib for plotting, and numpy for numerical operations.
#Read Data: It reads the dataset from the CSV file named 'height_weight_dataset.csv' using pandas read_csv() function and stores it in a DataFrame named df
