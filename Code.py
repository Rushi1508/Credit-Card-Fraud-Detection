import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv("D:/Review Paper/PS_20174392719_1491204439457_log.csv")  # Replace with your actual dataset path
data.head()

# Data Preprocessing
# Encode categorical features like "type" and "nameDest"
label_encoder = LabelEncoder()
data["type"] = label_encoder.fit_transform(data["type"])
data["nameDest"] = label_encoder.fit_transform(data["nameDest"])

# Define features and target variable
X = data.drop(["isFraud"], axis=1)
y = data["isFraud"]

# Check for missing values
missing_values = X.isnull().sum()
print(missing_values)

# Handle non-numeric columns (e.g., one-hot encoding)
non_numeric_cols = X.select_dtypes(exclude=['float64', 'int64']).columns
X_numeric = X.drop(non_numeric_cols, axis=1)  # Drop non-numeric columns
X_categorical = X[non_numeric_cols]  # Store non-numeric columns

# Standardize numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_numeric)

# Data Splitting
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Model Training
k = 3   # You can experiment with different values of k
knn_model = KNeighborsClassifier(n_neighbors=k)

# Find rows with any missing values
rows_with_missing = np.isnan(X_train).any(axis=1)

# Remove rows with missing values
X_train = X_train[~rows_with_missing]
y_train = y_train[~rows_with_missing]

# Adjust y_train accordingly
knn_model.fit(X_train, y_train)

# Model Prediction
y_pred = knn_model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
confusion_mat = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print("Confusion Matrix:\n", confusion_mat)
print("Classification Report:\n", classification_rep)

# Count the number of fraud and non-fraud instances
fraud_count = y.sum()
non_fraud_count = len(y) - fraud_count

# Create a pie chart
labels = ['Fraud', 'Not Fraud']
sizes = [fraud_count, non_fraud_count]
colors = ['red', 'green']
explode = (0.1, 0)  # Explode the 'Fraud' slice
plt.figure(figsize=(6, 6))
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
        shadow=True, startangle=140)
plt.axis('equal')  # Equal aspect ratio ensures that the pie is drawn as a circle.
plt.title('Fraud vs. Not Fraud Distribution')
plt.show()
