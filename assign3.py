# Assignment No. 3 - House Price Prediction
# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split  # Importing train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset
# Assuming your dataset is stored in a CSV file named 'housing_data.csv'
data = pd.read_csv('housing_data.csv')

# Define your independent variables (features) and dependent variable (target)
X = data[['area', 'bedrooms', 'bathrooms', 'stories']]
y = data['price']

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Calculate accuracy score (R-squared)
accuracy = r2_score(y_test, y_pred)
print("Accuracy Score:", accuracy)

# Calculate mean squared error
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Print coefficients and intercept
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

# Visualize features against price
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
sns.scatterplot(x='area', y='price', data=data)
plt.title('Area vs Price')

plt.subplot(2, 2, 2)
sns.scatterplot(x='bedrooms', y='price', data=data)
plt.title('Bedrooms vs Price')

plt.subplot(2, 2, 3)
sns.scatterplot(x='bathrooms', y='price', data=data)
plt.title('Bathrooms vs Price')

plt.subplot(2, 2, 4)
sns.scatterplot(x='stories', y='price', data=data)
plt.title('Stories vs Price')

plt.tight_layout()
plt.show()

# Plot actual price vs predicted price with regression line
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred)
sns.lineplot(x=y_test, y=y_test, color='red')  # Plotting the diagonal line (perfect prediction)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted Price with Regression Line')
plt.show()

# Take user input for prediction
area = float(input("Enter the area of the house: "))
bedrooms = int(input("Enter the number of bedrooms: "))
bathrooms = int(input("Enter the number of bathrooms: "))
stories = int(input("Enter the number of stories: "))

# Convert user input into a DataFrame
user_input = pd.DataFrame({'area': [area], 'bedrooms': [bedrooms], 'bathrooms': [bathrooms], 'stories': [stories]})

# Make prediction for user input
user_pred = model.predict(user_input)
print("Predicted price for the input:", user_pred[0])
