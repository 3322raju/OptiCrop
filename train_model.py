# ===============================
# OptiCrop - Smart Agricultural Production Optimization Engine
# Step 1: Import Required Libraries
# ===============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# ===============================
# Step 2: Read the Dataset
# ===============================

df = pd.read_csv("dataset/Crop_recommendation.csv")

# ===============================
# Step 3: Display the Dataset
# ===============================

# Display first 5 rows
print("========== First 5 Rows ==========")
print(df.head())

# Shape of the dataset
print("\n========== Shape ==========")
print(df.shape)

# Dataset information
print("\n========== Information ==========")
print(df.info())

# Statistical summary
print("\n========== Statistical Summary ==========")
print(df.describe())

# Check for missing values
print("\n========== Missing Values ==========")
print(df.isnull().sum())
# ===============================
# Distribution of Agricultural Features
# ===============================

plt.figure(figsize=(15,8))

plt.subplot(2,4,1)
sns.histplot(df['N'], kde=True, color='orange')
plt.title("Nitrogen")

plt.subplot(2,4,2)
sns.histplot(df['P'], kde=True, color='blue')
plt.title("Phosphorus")

plt.subplot(2,4,3)
sns.histplot(df['K'], kde=True, color='pink')
plt.title("Potassium")

plt.subplot(2,4,4)
sns.histplot(df['temperature'], kde=True, color='green')
plt.title("Temperature")

plt.subplot(2,4,5)
sns.histplot(df['humidity'], kde=True, color='purple')
plt.title("Humidity")

plt.subplot(2,4,6)
sns.histplot(df['ph'], kde=True, color='red')
plt.title("pH")

plt.subplot(2,4,7)
sns.histplot(df['rainfall'], kde=True, color='gold')
plt.title("Rainfall")

plt.tight_layout()
plt.show()
# ===============================
# Bivariate Analysis
# ===============================

plt.figure(figsize=(10,8))

sns.scatterplot(
    x='humidity',
    y='label',
    data=df
)

plt.title("Humidity vs Crop")
plt.xlabel("Humidity")
plt.ylabel("Crop")
plt.show()
# ===============================
# Outlier Detection
# ===============================

plt.figure(figsize=(15,8))

plt.subplot(2,4,1)
sns.boxplot(y=df['N'])
plt.title("Nitrogen")

plt.subplot(2,4,2)
sns.boxplot(y=df['P'])
plt.title("Phosphorus")

plt.subplot(2,4,3)
sns.boxplot(y=df['K'])
plt.title("Potassium")

plt.subplot(2,4,4)
sns.boxplot(y=df['temperature'])
plt.title("Temperature")

plt.subplot(2,4,5)
sns.boxplot(y=df['humidity'])
plt.title("Humidity")

plt.subplot(2,4,6)
sns.boxplot(y=df['ph'])
plt.title("pH")

plt.subplot(2,4,7)
sns.boxplot(y=df['rainfall'])
plt.title("Rainfall")

plt.tight_layout()
plt.show()
# ===============================
# Detect Outliers using IQR
# ===============================

Q1 = df['K'].quantile(0.25)
Q3 = df['K'].quantile(0.75)

IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

print("\n========== Outlier Detection ==========")
print("Lower Bound :", lower)
print("Upper Bound :", upper)

outliers = df[(df['K'] < lower) | (df['K'] > upper)]

print("Number of Outliers :", len(outliers))
# ===============================
# Feature Selection
# ===============================

X = df.drop("label", axis=1)

y = df["label"]

print("\n========== Features ==========")
print(X.head())

print("\n========== Target ==========")
print(y.head())
# ===============================
# Train Test Split
# ===============================

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = df.drop("label", axis=1)
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Feature Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("\nTraining Data :", X_train.shape)
print("Testing Data :", X_test.shape)
# ===============================
# Logistic Regression Model
# ===============================

from sklearn.linear_model import LogisticRegression

# Create the model
lr_model = LogisticRegression(max_iter=1000)

# Train the model
lr_model.fit(X_train, y_train)

print("\n========== Model Training Completed ==========")
# ===============================
# Prediction
# ===============================

y_pred = lr_model.predict(X_test)

print("\n========== Sample Predictions ==========")

for i in range(10):
    print("Actual:", y_test.iloc[i], " | Predicted:", y_pred[i])
    # ===============================
# Model Accuracy
# ===============================

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)

print("\n========== Model Accuracy ==========")
print("Accuracy:", accuracy * 100, "%")
# ===============================
# Classification Report
# ===============================

from sklearn.metrics import classification_report

print("\n========== Classification Report ==========")

print(classification_report(y_test, y_pred))
# ===============================
# Confusion Matrix
# ===============================

from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10,8))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()
# ===============================
# Save Model
# ===============================

import pickle

with open("models/model.pkl", "wb") as file:
    pickle.dump(lr_model, file)

print("\nModel saved successfully!")