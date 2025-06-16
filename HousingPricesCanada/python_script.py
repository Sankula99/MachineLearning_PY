import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from math import sqrt
from xgboost import XGBClassifier, XGBRegressor
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
# Load the California housing dataset
from sklearn.datasets import fetch_california_housing 
file_path =r"File Location on the PC"
data = pd.read_csv(file_path,header='infer', na_values='?')
print(data.head(10))
data.info()
#Check for unique values in the 'ocean_proximity' column
print(data["ocean_proximity"].unique())

#Data Preprocessing and Cleaning

#Check for missing values
missing_values = data.isnull().sum()
print("Missing values in each column:\n", missing_values)

#Calculate the percentage of missing values
missing_percentage = (missing_values / len(data)) * 100
print("\nPercentage of missing values in each column:\n", missing_percentage)

#Remove rows with missing values
dataCleaned = data.dropna()

#Verify that there are no missing values left
missing_values_after = dataCleaned.isnull().sum()  
print("\nMissing values after dropping rows:\n", missing_values_after)


#Data Exploration and Visualization
print(dataCleaned.describe())

#sns.set(style="whitegrid")
#plt.figure(figsize=(10, 6))
#sns.histplot(dataCleaned['median_house_value'],color='red', kde=True)
#plt.title('Distribution of Median House Value')
#plt.xlabel('Median House Value')
#plt.ylabel('Frequency')
#plt.show()


#Remove outliers in 'median_house_value' based on Interquartile Range (IQR)
Q1 = dataCleaned['median_house_value'].quantile(0.25)
Q3 = dataCleaned['median_house_value'].quantile(0.75)
IQR = Q3 - Q1

#Define lower and upper bounds for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

datawithout_outliers = dataCleaned[(dataCleaned['median_house_value'] >= lower_bound) & (dataCleaned['median_house_value'] <= upper_bound)]
print("New Data shape without outliers",datawithout_outliers.shape)


#Another Approach to visualize outliers using boxplot
#plt.figure(figsize=(10, 6))
#sns.boxplot(x=datawithout_outliers['median_income'],color='red')
#plt.title('Boxplot of Median Income') 
#plt.xlabel('Median Income')
#plt.show()

#Remove outliers in 'median_income' based on Interquartile Range (IQR)
Q1_mi = datawithout_outliers['median_income'].quantile(0.25)
Q3_mi = datawithout_outliers['median_income'].quantile(0.75)
IQR_mi = Q3_mi - Q1_mi

#Define lower and upper bounds for outliers
lower_bound_mi = Q1_mi - 1.5 * IQR_mi
upper_bound_mi = Q3_mi + 1.5 * IQR_mi

datawithout_outliers_mi = datawithout_outliers[(datawithout_outliers['median_income'] >= lower_bound_mi) & (datawithout_outliers['median_income'] <= upper_bound_mi)]
#print("New Data shape without outliers",datawithout_outliers_mi.shape)
#plt.figure(figsize=(10, 6))
#sns.boxplot(x=datawithout_outliers_mi['median_income'],color='orange')
#plt.title('Boxplot of Median Income without Outliers') 
#plt.xlabel('Median Income')
#plt.show()

#Correlation Analysis
#plt.figure(figsize=(12, 8))
#sns.heatmap(data.select_dtypes(include='number'), annot=True, cmap='coolwarm',linewidths=0.5)
#plt.title('Correlation Heatmap')
#plt.show()

#Define the independent(everything else ) and dependent variables (medina_house_value)
data_encoded = pd.get_dummies(datawithout_outliers_mi, columns=['ocean_proximity'])

features = ['longitude', 'latitude', 'housing_median_age', 'total_rooms',
           'total_bedrooms', 'population', 'households', 'median_income', 
           'ocean_proximity_NEAR BAY', 'ocean_proximity_INLAND',
           'ocean_proximity_ISLAND', 'ocean_proximity_NEAR OCEAN',
           'ocean_proximity_<1H OCEAN']
target = 'median_house_value'
x = data_encoded[features]
y=data_encoded[target]

#Split the data for training and testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1111)
print(f'Training data shape: {x_train.shape[0]} samples')
print(f'Testing data shape: {x_test.shape[0]} samples')

x_train=x_train.astype(float)
x_test=x_test.astype(float)
#Training the Linear Regression Model
X_Train_const = sm.add_constant(x_train)  # Add a constant term for the intercept
print(f"Shape of X_Train_Model:\n", X_Train_const)


#Fit the OLS model
model_fitted =sm.OLS(y_train,X_Train_const).fit()

#Print the model summary
print(model_fitted.summary())

#Make predictions on the test set
X_Test_const = sm.add_constant(x_test)  # Add a constant term for the intercept
print("---------------------------------------")
print("---------------------------------------")
test_predictions = model_fitted.predict(X_Test_const)
print("Test Predictions:\n", test_predictions.head())

#Checking Assumptions of Linear Regression
#Linearity
#plt.scatter(y_test,test_predictions, color='blue', alpha=0.5)
#plt.xlabel('Observed Values')
#plt.ylabel('Predicted Values')
#plt.plot(y_test,y_test, color='red', linewidth=2)  # Diagonal line for reference
#plt.show()

#Random Sample
mean_residuals = np.mean(model_fitted.resid)
print("Mean of Residuals:", {np.round(mean_residuals,2)})

#Exogenity
residuals = model_fitted.resid
#Check for correlation between residuals and each predictor
for column in x_train.columns:
    correlation = np.corrcoef(x_train[column], residuals)[0, 1]
    print(f"Correlation between residuals and {column}: {np.round(correlation, 2)}")

#Homoskedasticity
#plt.scatter(model_fitted.fittedvalues, model_fitted.resid, color='green', alpha=0.5)
#plt.axhline(0, color='red', linestyle='--')
#plt.xlabel('Fitted Values')
#plt.ylabel('Residuals')
#plt.title('Residuals vs Fitted Values')
#plt.show()

#Scaling the Data
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

#Crate and fit the model 
lr = LinearRegression()
lr.fit(x_train_scaled, y_train)
# Make predictions on the test set
y_pred_scaled = lr.predict(x_test_scaled)
# Calculate MSE and RMSE
mse_scaled = mean_squared_error(y_test, y_pred_scaled)
rmse_scaled = sqrt(mse_scaled)
print(f"Scaled Model MSE: {mse_scaled}")
print(f"Scaled Model RMSE: {rmse_scaled}")

modelXGB = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
modelXGB.fit(x_train_scaled, y_train)
y_predXGB =modelXGB.predict(x_test_scaled)
# Calculate MSE and RMSE for XGB model  
mse_XGB = mean_squared_error(y_test, y_predXGB)
rmse_XGB = sqrt(mse_XGB)
print(f"XGB Model RMSE: {rmse_XGB}")

#Random Forest Regressor
modelRF = RandomForestRegressor(n_estimators=100,max_depth=10 ,random_state=42)
modelRF.fit(x_train_scaled, y_train)
y_predRF = modelRF.predict(x_test_scaled)
mse_RF = mean_squared_error(y_test, y_predRF)
rmse_RF = sqrt(mse_RF)
print(f"Random Forest Model RMSE: {rmse_RF}")
#Decision Tree Regressor
modelDT = DecisionTreeRegressor(max_depth=10, random_state=42)
modelDT.fit(x_train_scaled, y_train)
y_predDT = modelDT.predict(x_test_scaled)
mse_DT = mean_squared_error(y_test, y_predDT)
rmse_DT = sqrt(mse_DT)
print(f"Decision Tree Model RMSE: {rmse_DT}")
