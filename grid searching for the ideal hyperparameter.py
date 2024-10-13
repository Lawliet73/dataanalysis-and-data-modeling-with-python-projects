from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
import numpy as np
from sklearn.metrics import r2_score

# After selecting a model (in this case, a polynomial of order 3) to capture the complex relationship 
# in our data, we must consider how well this model generalizes to unseen future data. To address this, 
# we focus on controlling the model's generalization error. A model that fits the training data too 
# closely may overfit and perform poorly on new data. 

# To prevent overfitting, we use Ridge regression, which applies L2 regularization to penalize large 
# coefficients and reduce model complexity. This helps the model generalize better to unseen data.

# To further optimize the model, we implement a Grid Search with cross-validation. The goal is to find 
# the optimal value of the regularization parameter alpha (which controls the strength of regularization). 
# During this process, the model is trained on a training set and evaluated on a validation set using 
# cross-validation. Grid Search tests different values of alpha to find the best one that balances bias 
# and variance in the model.

# Finally, after determining the best alpha through cross-validation, we retrain the model on the full 
# training set and evaluate its performance on a separate test set. This allows us to assess how well 
# the model generalizes to new, unseen data.




# Sample data with a complex relationship (cubic + noise)
np.random.seed(0)  # For reproducibility
X = np.array([[100], [150], [200], [250], [300], [350], [400], [450], [500], [550], [600]])
y = 20000 + 5*(X**3) - 2*(X**2) + 100*X + np.random.randn(*X.shape)*100000

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the hyperparameters (alpha) to test
param_grid = {'ridge__alpha': [0.001, 0.01, 0.1, 1, 10]}

# Create a pipeline with scaling, polynomial features, and Ridge regression
pipeline = make_pipeline(StandardScaler(), PolynomialFeatures(degree=3), Ridge())

# Grid Search with cross-validation on the training data
grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=3, scoring='r2')

# Train model with Grid Search on the training set
grid_search.fit(X_train, y_train)

# Output the best hyperparameters and R² score from the cross-validation
best_params = grid_search.best_params_
best_cv_score = grid_search.best_score_

print(f"Best Hyperparameters: {best_params}")
print(f"Best Cross-Validation R² Score: {best_cv_score}")

# Once the best hyperparameters are found, retrain the model on the full training data
final_model = grid_search.best_estimator_
final_model.fit(X_train, y_train)

# Evaluate the final model on the test set (unseen data)
y_pred = final_model.predict(X_test)
test_r2_score = r2_score(y_test, y_pred)

print(f"Test R² Score: {test_r2_score}")