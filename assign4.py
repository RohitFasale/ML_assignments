#Assignment No. 4 - BankNote Authentication

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, r2_score

url = 'BankNote_Authentication.csv'
df = pd.read_csv(url, header=0)  # Specify header=0 to treat the first row as the column names

x = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

classifier = LogisticRegression(random_state=0)
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

cm = confusion_matrix(y_test, y_pred)
print(cm)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy : ", accuracy)

# Take input for prediction
variance = float(input("Enter variance: "))
skewness = float(input("Enter skewness: "))
curtosis = float(input("Enter curtosis: "))
entropy = float(input("Enter entropy: "))

# Preprocess input
input_data = [[variance, skewness, curtosis, entropy]]

# Make prediction
predicted_authentication = classifier.predict(input_data)

# Display prediction
print("Predicted Authentication:", predicted_authentication[0])