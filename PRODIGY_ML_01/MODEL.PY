import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


# Load dataset
df = pd.read_csv('train.csv')

# View basic info
df[['GrLivArea', 'BedroomAbvGr', 'FullBath', 'SalePrice']].head()



# Select relevant columns
df = df[['GrLivArea', 'BedroomAbvGr', 'FullBath', 'SalePrice']]

# Check for missing values
print(df.isnull().sum())

# Drop rows with missing values if any
df.dropna(inplace=True)



# Features and target
X = df[['GrLivArea', 'BedroomAbvGr', 'FullBath']]  # Independent variables
y = df['SalePrice']                                 # Target variable

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



model = LinearRegression()
model.fit(X_train, y_train)



# Predict
y_pred = model.predict(X_test)

# Metrics
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R-squared Score:", r2_score(y_test, y_pred))

# Model coefficients
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)

# Optional: visualize prediction vs actual
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")
plt.show()
