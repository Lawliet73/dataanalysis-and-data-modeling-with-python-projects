import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.linear_model import LinearRegression,Ridge
from sklearn.metrics import r2_score, mean_squared_error
from scipy import stats
from sklearn.model_selection import GridSearchCV,train_test_split


import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")


#read csv files using pandas and load  into a dataframe(eg. df=dataframe)
df= pd.read_csv("kc_house_data_NaN.csv")


#check if read and converted correctly
print(df.head(),"\n\n " )
#check if coulmns hold the correct data types 
print("data types ",df.dtypes,"\n\n ")
#getting info on the data
print("Data Info:", df.info())

#Data cleaning/wrangling
#checking for duplicates
duplicates= df[df.duplicated()]
if not duplicates.empty:  
    df = df.drop_duplicates()  # Drop the duplicates
    print("\nDuplicates have been dropped.")
else:
    print("\nNo duplicates found.\n")

# Check for missing values
missing_values = df.isnull().sum()

# dropping missing values 
#if missing_values.sum() > 0:  # Check if there are any missing values (non-zero sum)
    #print("\n",missing_values,"\n")
    #df = df.dropna()    Drop the rows with missing values if needed
    #print("\nMissing values have been found.")else:print("\nNo missing values found.")


#added after noticing missing values for columns bedrooms and bathrooms to replace them
#with the average of their column
mean=df['bedrooms'].mean()
df['bedrooms'].replace(np.nan,mean, inplace=True)
mean=df['bathrooms'].mean()
df['bathrooms'].replace(np.nan,mean, inplace=True)


#unnecessary columns dropped
df = df.drop("id", axis=1)
df = df.drop("Unnamed: 0", axis=1)

#editable summarized data overview
print("\n\n\n\nsummarized table content \n\n",df.head()[['date', 'price', 'bedrooms',"sqft_living",'sqft_above']])

#get a quick overview of the data before starting to analyze
print("\n\nStatistical summary\n\n", df.describe(include="all"),"\n\n")

#finding out how many houses have a unique amount of floors
floor_counts = df['floors'].value_counts()
floor_counts_df = floor_counts.to_frame(name='houses with X floors').reset_index()
floor_counts_df = floor_counts_df.rename(columns={'index': 'floors'})
print(floor_counts_df)

#correlation between price and a waterfront view
df_waterview_pricing= df[["price","view","waterfront"]]
print("\n\n df_waterview_pricing: \n\n ",df_waterview_pricing)
sns.boxplot(x="waterfront", y="price", data=df)
plt.show()


#investigating correlation between sqft and price 
sns.regplot(x="sqft_above",y="price",data=df)
plt.ylim(0)
plt.show()



# .corr() computes pairwise correlation of columns and returns features correlated to price
# as a correlation matrix. This can help determine which other factors closely relate to
#the price of the houses

# Select only numeric columns for both correlation and p-value calculations
df_numeric = df.select_dtypes(include=[float, int])
# Calculate the correlation matrix for numeric columns and sort by "price"
correlation_matrix = df_numeric.corr()["price"].sort_values()
# Filter correlations to include only those greater than 
high_correlations = correlation_matrix[correlation_matrix > 0.7]
# Convert the correlation matrix to a DataFrame(only for better output format)
high_correlations_df = high_correlations.to_frame(name='Correlation')
# Print the columns with high correlation to price
print("\n\n Price determining factors:\n\n",high_correlations_df)

# Check p-values of the correlations to evaluate certainty
p_values = []
# Loop over the columns in the high correlations and calculate their p-values
for col in high_correlations.index:
        _, p_value = stats.pearsonr(df_numeric[col], df_numeric["price"])
        p_values.append((col, p_value))
# Print the columns and their corresponding p-values
for col, p_value in p_values:
    print("\n\n",f"Column: {col}, P-value: {p_value}","\n\n")





# model development: simple lineal regression of sqft Area and price

X = df[['sqft_living']].values.reshape(-1, 1)  # Reshape X to be 2D (n_samples, 1 feature)
Y = df['price'].values                         # Convert Y to a NumPy array (1D array)
lm = LinearRegression()
lm.fit(X,Y)
y_pred_simple= lm.predict(X)  # evaluates how well model represents provided/training data

print(f"R² score: {lm.score(X, Y)}")
plt.scatter(X.flatten(),Y, color="blue", label="Actual data")
plt.plot(X.flatten(), y_pred_simple, color="red", label="Regression line")
plt.xlabel("sqft Area")
plt.ylabel("Price")
plt.title("Simple Linear Regression: sqft Area vs Price")
plt.legend()
plt.show()





# model development: multiple lineal regression of features possibly correlating to price

features = ["floors", "waterfront", "lat", "bedrooms", "sqft_basement", 
            "view", "bathrooms", "sqft_living15", "sqft_above", "grade", "sqft_living"]
X = df[features].values  # Multiple features, no need to reshape in this case
Y = df['price'].values   

lm = LinearRegression()
lm.fit(X, Y)
y_pred = lm.predict(X)
print("\n\n",f"R² score MLR: {lm.score(X, Y)}", "\n\n")

# Plotting is not ideal for multiple features, so we just evaluate model performance
# Instead plot actual vs predicted prices in a scatter plot to evaluate,in an ideal 
#model this results in scatter plot resembling the identity function

plt.scatter(Y, y_pred, color="blue", label="Predicted vs Actual Prices")
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title(f"Multiple Linear Regression:Actual vs Predicted Prices (R²={lm.score(X, Y):.2f})")
plt.legend()
plt.show()







#pipeline
pipeline = Pipeline([("scaler",StandardScaler()),
                     ("polynomial", PolynomialFeatures(include_bias=False,degree=2)),
                     ("model",LinearRegression())])

pipeline.fit(X, Y)
y_pred = pipeline.predict(X)
r2 = r2_score(Y, y_pred)
mse = mean_squared_error(Y,y_pred)
print(f"R² score: {r2}")
print(f"Mean Squared Error: {mse}", "\n\n")





#ridge regression

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=1)

print("\n\n number of test samples:\n\n", x_test.shape[0],"\n\n")
print("number of training samples:\n\n",x_train.shape[0],"\n\n")

# hyperparameters (alpha) to test
param_grid = {'ridge__alpha': [0.001, 0.01, 0.1, 1, 10]}

# pipeline with scaling, polynomial features, and Ridge regression
pipeline2 = make_pipeline(StandardScaler(), PolynomialFeatures(degree=3), Ridge())

# Grid Search with cross-validation,train model and Output the best hyperparameters
grid_search = GridSearchCV(estimator=pipeline2, param_grid=param_grid, cv=3, scoring='r2')
grid_search.fit(x_train, y_train)
best_params = grid_search.best_params_
best_cv_score = grid_search.best_score_
print(f"Best Hyperparameters: {best_params}","\n\n")
print(f"Best Cross-Validation R² Score: {best_cv_score}","\n\n")

# Once the best hyperparameters are found, retrain the model on the full training data
final_model = grid_search.best_estimator_
final_model.fit(x_train, y_train)

# Evaluate the final model on the test set (unseen data)
y_pred = final_model.predict(x_test)
test_r2_score = r2_score(y_test, y_pred)
print(f"Test R² Score Ridge regression: {test_r2_score}","\n\n")


# custom Ridge evaluation with alpha=0.1 (no polynomial transformation)

ridge = Ridge(alpha=0.1)
ridge.fit(x_train, y_train)
y_pred = ridge.predict(x_test)
r2 = r2_score(y_test, y_pred)

print(f"Test R² Score custom Ridge regression 1: {r2}","\n\n")



#custom ridge evaluation alpha=0.1 and PolynomialFeatures(degree=2) 
poly = PolynomialFeatures(degree=2)
x_train_poly= poly.fit_transform(x_train)
x_test_poly = poly.transform(x_test)

alpha_input = input("Enter alpha value: ")  
if len(alpha_input) < 1:  # If input is empty, set default value
    alpha_input = 0.1
else:
    alpha_input = float(alpha_input)  # if given input Convert input to float

ridge = Ridge(alpha=alpha_input)


ridge.fit(x_train_poly, y_train)      # train model on the polynomial-transformed dataa
custom_ridge_r2score= r2_score(y_test, ridge.predict(x_test_poly))
print(f"Test R² Score Custom Ridge regression 2: {custom_ridge_r2score}","\n\n")







