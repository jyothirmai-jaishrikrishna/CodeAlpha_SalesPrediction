# CodeAlpha Task 4 - Sales Prediction Using Python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# 1. Load the dataset
df = pd.read_csv("advertising_sales.csv")

print("First 5 rows:")
print(df.head())


# 2. Check the dataset
print("\nDataset Shape:")
print(df.shape)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())


# 3. Remove unnecessary index column
if "Unnamed: 0" in df.columns:
    df = df.drop("Unnamed: 0", axis=1)

# Remove duplicate rows
df = df.drop_duplicates()

print("\nCleaned Dataset:")
print(df.head())

print("\nCleaned Shape:")
print(df.shape)


# 4. Basic statistics
print("\nDescriptive Statistics:")
print(df.describe())


# 5. Correlation analysis
print("\nCorrelation with Sales:")
print(df.corr(numeric_only=True)["Sales"].sort_values(ascending=False))


# 6. Sales Distribution
plt.figure(figsize=(8, 5))

plt.hist(df["Sales"], bins=20)

plt.title("Sales Distribution")
plt.xlabel("Sales")
plt.ylabel("Frequency")

plt.tight_layout()
plt.savefig("sales_distribution.png")
plt.show()


# 7. Advertising vs Sales
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].scatter(df["TV"], df["Sales"])
axes[0].set_title("TV Advertising vs Sales")
axes[0].set_xlabel("TV Advertising")
axes[0].set_ylabel("Sales")

axes[1].scatter(df["Radio"], df["Sales"])
axes[1].set_title("Radio Advertising vs Sales")
axes[1].set_xlabel("Radio Advertising")
axes[1].set_ylabel("Sales")

axes[2].scatter(df["Newspaper"], df["Sales"])
axes[2].set_title("Newspaper Advertising vs Sales")
axes[2].set_xlabel("Newspaper Advertising")
axes[2].set_ylabel("Sales")

plt.tight_layout()
plt.savefig("advertising_vs_sales.png")
plt.show()


# 8. Correlation Heatmap
plt.figure(figsize=(8, 6))

sns.heatmap(
    df.corr(numeric_only=True),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")

plt.tight_layout()
plt.savefig("correlation_heatmap.png")
plt.show()


# 9. Features and target
X = df[["TV", "Radio", "Newspaper"]]
y = df["Sales"]


# 10. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)


# 11. Train Linear Regression model
model = LinearRegression()

model.fit(X_train, y_train)


# 12. Make predictions
y_pred = model.predict(X_test)


# 13. Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n----- Model Evaluation -----")

print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("R² Score:", r2)


# 14. Advertising coefficients
coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})

print("\n----- Advertising Coefficients -----")
print(coefficients)


# 15. Actual vs Predicted Sales
plt.figure(figsize=(8, 6))

plt.scatter(y_test, y_pred)

plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")

plt.tight_layout()
plt.savefig("actual_vs_predicted.png")
plt.show()


# 16. Advertising channel contribution
plt.figure(figsize=(8, 5))

plt.bar(
    coefficients["Feature"],
    coefficients["Coefficient"]
)

plt.title("Advertising Channel Contribution")
plt.xlabel("Advertising Channel")
plt.ylabel("Regression Coefficient")

plt.tight_layout()
plt.savefig("feature_importance.png")
plt.show()


# 17. Example sales prediction
example = pd.DataFrame({
    "TV": [150],
    "Radio": [30],
    "Newspaper": [20]
})

predicted_sales = model.predict(example)

print("\n----- Example Prediction -----")

print("TV Advertising:", 150)
print("Radio Advertising:", 30)
print("Newspaper Advertising:", 20)

print("Predicted Sales:", predicted_sales[0])