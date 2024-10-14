
# King County Housing Price Prediction Project

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset Information](#Dataset-Information)
- [Dependencies](#Dependencies)
- [Steps and Code Explanation](#Steps-and-Code-Explanation)
- [How to Run the Code](#How-to-Run-the-Code)
- [Outputs and Results](#Outputs-and-Results)
- [Author](#Author)
  
## Project Overview

This project involves analyzing and predicting housing prices in King County, Washington (which includes Seattle) using various machine learning techniques. The dataset contains house sale prices for homes sold between May 2014 and May 2015. As a Data Analyst working at a Real Estate Investment Trust (REIT), your task is to determine the market price of a house given a set of features such as square footage, number of bedrooms, number of floors, and more.

The project utilizes several models to predict house prices based on these features:

- Simple and Multiple Linear Regression
- Polynomial Regression
- Ridge Regression (with and without polynomial transformation)
  
Additionally, hyperparameter tuning is performed using Grid Search to optimize the model's performance.

## Dataset Information

The dataset used in this project is kc_house_data_NaN.csv and contains the following key columns:

- price: The sale price of the house (target variable).
- bedrooms: Number of bedrooms.
- bathrooms: Number of bathrooms.
- sqft_living: Square footage of the living area.
- sqft_above: Square footage of the house, excluding the basement.
- floors: Number of floors in the house.
- waterfront: Whether the house has a waterfront view.
...and several other features.

## Dependencies

To run this project, you will need the following Python libraries:

- pandas: For data loading, manipulation, and cleaning.
- matplotlib & seaborn: For data visualization.
- scikit-learn: For model building, training, and evaluation (including linear regression, Ridge regression, polynomial features, and GridSearchCV).
- numpy: For numerical computations.
- scipy: For statistical computations (Pearson correlation and p-values).


## Steps and Code Explanation
### 1. Data Loading and Cleaning

- The dataset is loaded using pandas.read_csv().
- Data cleaning includes checking for duplicates, handling missing values, and dropping unnecessary columns (id and Unnamed: 0).
- Missing values in bedrooms and bathrooms are replaced with the column mean.
  
### 2. Exploratory Data Analysis (EDA)

- Basic statistics and an overview of the dataset are obtained using df.describe() and df.info().
- Correlations between features (e.g., square footage and price) are explored using seaborn visualizations and the Pearson correlation coefficient.
- Visualizations such as boxplots and scatterplots show the relationship between house prices and features like waterfront and sqft_living.
  
### 3. Model Development

- Simple Linear Regression: A model is trained to predict house prices based on sqft_living alone. The R² score is calculated to assess the performance.

- Multiple Linear Regression: A model is built using multiple features (e.g., floors, bedrooms, bathrooms) to predict house prices. The R² score is evaluated.

- Polynomial Regression: Polynomial features (degree 2) are generated to capture non-linear relationships between features and house prices. A pipeline is used for standardization and model fitting.

### 4. Ridge Regression with Hyperparameter Tuning

- Grid Search with Ridge Regression: A pipeline is built with scaling, polynomial features (degree 3), and Ridge regression. Hyperparameter tuning with GridSearchCV is performed to find the best alpha using cross-validation. The best hyperparameters and the cross-validation R² score are displayed.
  
### 5. Custom Ridge Regression Implementation

- Custom Ridge Regression without Polynomial Features: A Ridge regression model with alpha=0.1 is trained and evaluated on the test set.

- Custom Ridge Regression with Polynomial Features: A second-order polynomial transformation is applied to the training and test data, followed by Ridge regression (alpha=0.1), and the R² score is evaluated.

## How to Run the Code

1. Ensure the required Python libraries are installed.
2. Place the dataset kc_house_data_NaN.csv in the same directory as the script, or update the file path in the code.
3. Run the Python script to load, clean, and analyze the data, and to generate the predictive models.
4. The output will include R² scores and visualizations that show relationships between house features and prices.


## Outputs and Results

- R² Scores: R² scores are calculated for each model to assess how well the model predicts house prices.
- Visualizations: Scatter plots, regression plots, and box plots illustrate the relationship between house prices and the features.
- Hyperparameter Tuning: Grid Search is used to find the optimal Ridge regularization parameter (alpha).

## Additional Programs: Finding the Best Polynomial Degree

### 1. Program to Find the Best Polynomial Degree Based on R²
This program evaluates different polynomial degrees to determine which degree best fits the data based on the R² score for the test set.

***Steps***:
- Train the model on different polynomial degrees (e.g., degrees 1 to 5).
- Evaluate the R² score on the test set for each degree.
- Find the degree that maximizes the R² score.
  
### 2. Program to Visualize Polynomial Degree Performance
This program applies Ridge regression for different polynomial degrees and visualizes the R² scores for both training and testing sets. The graph helps in selecting the best degree based on the R² score.

***Steps***:
- Generate synthetic data (sinusoidal with noise).
- Apply polynomial transformations of varying degrees (e.g., degrees 1 to 15).
- Fit a Ridge regression model for each polynomial degree.
- Plot the R² scores for both the training and test sets to visualize model performance.

### 3. How to Use the Programs for Model Selection and Hyperparameter Tuning:

Once you have determined the best polynomial degree using either R² scores or visualizations, you can proceed to refine the model further by applying Ridge regression and cross-validation. The next step involves selecting the best regularization hyperparameter (alpha) using GridSearchCV.

***Steps for Model Selection***:

- Run a modified version of the programs to find the best polynomial degree by evaluating R² scores or visualizing the performance across different degrees.
- Select the best degree based on the highest R² score or the degree that balances performance on both training and test sets (avoiding overfitting or underfitting).
- After this the Ridge regression with cross-validation (e.g., GridSearchCV) in the main code can be applied to find the best regularization parameter (alpha) for the selected polynomial degree. This will help you fine-tune the model to prevent overfitting and improve generalization to unseen data.
  
By using these two programs in conjunction, you ensure that the model not only fits the data well but also generalizes effectively, as Ridge regression helps control the complexity of the polynomial model by penalizing large coefficients.


## Final Notes

This project demonstrates the process of predicting housing prices in King County using a variety of machine learning models, from simple linear regression to more advanced techniques like Ridge regression with polynomial features and hyperparameter tuning. The code is designed to help capture the relationship between various house features and their prices in a comprehensive manner.

## Author

Kalab Alemayehu
