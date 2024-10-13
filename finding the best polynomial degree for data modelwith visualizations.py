import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")

# 1. Create a synthetic dataset
np.random.seed(0)
X = np.sort(np.random.rand(100, 1) * 10, axis=0)  # Random data points
y = np.sin(X).ravel() + np.random.randn(100) * 0.5  # Sinusoidal with noise

# 2. Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Define range of polynomial orders
degrees = np.arange(1, 16)

# Arrays to store R² scores
train_r2_scores = []
test_r2_scores = []

# 4. Apply Ridge Regression for each polynomial degree
for degree in degrees:
    model = make_pipeline(PolynomialFeatures(degree), Ridge(alpha=1))
    model.fit(X_train, y_train)
    
    # Predict on training and test data
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate R²
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    # Store the R² scores
    train_r2_scores.append(train_r2)
    test_r2_scores.append(test_r2)

# 5. Plotting the R² scores for training and test sets
plt.figure(figsize=(8, 6))
plt.plot(degrees, train_r2_scores, label='Training R²', color='red')
plt.plot(degrees, test_r2_scores, label='Test R²', color='blue')
plt.axvline(x=degrees[np.argmax(test_r2_scores)], linestyle='--', color='purple', label='Best Order')
plt.xlabel('Polynomial Degree (Order)')
plt.ylabel('R² Score')
plt.title('Ridge Regression: Model Selection Based on Polynomial Degree (R²)')
plt.legend()
plt.show()