from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import numpy as np


# Sample data (Horsepower and Price)
X_train = np.array([100, 150, 200, 250, 300]).reshape(-1, 1)  # Horsepower
y_train = np.array([20000, 25000, 30000, 35000, 40000])       # Price

X_test = np.array([120, 180, 240, 280]).reshape(-1, 1)        # Test Horsepower
y_test = np.array([21000, 27000, 32000, 37000])               # Test Price

# Trying different polynomial degrees
degrees = [1, 2, 3, 4, 5]
r2_scores = []

for degree in degrees:
    poly = PolynomialFeatures(degree=degree)
    X_poly_train = poly.fit_transform(X_train)
    X_poly_test = poly.transform(X_test)
    
    model = LinearRegression()
    model.fit(X_poly_train, y_train)
    
    # Predictions and R^2 for test data
    y_pred_test = model.predict(X_poly_test)
    r2 = r2_score(y_test, y_pred_test)
    r2_scores.append(r2)
    
    print(f"Degree: {degree}, Test R^2: {r2}")

# Find the best polynomial degree based on R^2
best_degree = degrees[np.argmax(r2_scores)]
print(f"Best Polynomial Degree: {best_degree}")