**Definition:**  
A Random Forest is an ensemble learning method for classification, regression, and other tasks  
that operates by constructing a multitude of decision trees during training

**Purpose:**  
It addresses the limitations of single decision trees by reducing overfitting and improving  
accuracy.  
- Random forests combine the simplicity of decision trees with the power of ensemble learning.  
- They are versatile, applicable to various types of problems including classification and regression.

# How they work 

![[Pasted image 20241014140943.png]]
## Key Hyperparameters
![[Pasted image 20241014141437.png]]

# Example of Random Forest 

## Example Melbourne Housing Snapshot

### Context

Melbourne real estate is BOOMING. Can you find the insight or predict the next big trend to become a real estate mogul… or even harder, to snap up a reasonably priced 2-bedroom unit?

### Content

This is a snapshot of a [dataset created by Tony Pino](https://www.kaggle.com/anthonypino/melbourne-housing-market).

It was scraped from publicly available results posted every week from [Domain.com.au](http://domain.com.au/). He cleaned it well, and now it's up to you to make data analysis magic. The dataset includes Address, Type of Real estate, Suburb, Method of Selling, Rooms, Price, Real Estate Agent, Date of Sale and distance from C.B.D.

### Notes on Specific Variables

**Rooms**: Number of rooms

**Price**: Price in dollars

**Method**: S - property sold; SP - property sold prior; PI - property passed in; PN - sold prior not disclosed; SN - sold not disclosed; NB - no bid; VB - vendor bid; W - withdrawn prior to auction; SA - sold after auction; SS - sold after auction price not disclosed. N/A - price or highest bid not available.

**Type**: br - bedroom(s); h - house,cottage,villa, semi,terrace; u - unit, duplex; t - townhouse; dev site - development site; o res - other residential.

**SellerG**: Real Estate Agent

**Date**: Date sold

**Distance**: Distance from CBD

**Regionname**: General Region (West, North West, North, North east …etc)

**Propertycount**: Number of properties that exist in the suburb.

**Bedroom2** : Scraped # of Bedrooms (from different source)

**Bathroom**: Number of Bathrooms

**Car**: Number of carspots

**Landsize**: Land Size

**BuildingArea**: Building Size

**CouncilArea: Governing council for the area

## Initial Data Exploration

```python
# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.tree import export_graphviz
import graphviz

# import the dataset
df = pd.read_csv('melb_data.csv')

# Converting column names to lower case
df.columns = df.columns.str.lower()

# Displaying basic information about the DataFrame
df.info()

# Displaying descriptive statistics of the DataFrame
df.describe()

# Checking for missing data
df.isnull().sum()

# Visualizing the distribution of prices
plt.figure(figsize=(10, 6))
sns.histplot(df['price'], kde=True)
plt.title('Distribution of Prices')
plt.show()
```

![[Pasted image 20241014142319.png]]
```python
# Visualizing the correlation matrix
plt.figure(figsize=(14, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()
```
![[Pasted image 20241014142343.png]]
```python
# Deep dive into 'Rooms' vs 'Price'
plt.figure(figsize=(8, 6))
sns.boxplot(x='rooms', y='price', data=df)
plt.title('Rooms vs Price')
plt.show()
```
![[Pasted image 20241014142403.png]]

# Model Training
```python

# Preparing the data for modeling
X = df.drop(columns=['price', 'date', 'address'])
y = df['price']

# Encoding categorical variables
X = pd.get_dummies(X, drop_first=True)

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating and training the Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Making predictions
y_pred = model.predict(X_test)

# Evaluating the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f'Mean Absolute Error: {mae}')
print(f'Mean Squared Error: {mse}')
print(f'Root Mean Squared Error: {rmse}')

# Example usage:
mape_score = mean_absolute_percentage_error(y_test, y_pred)
print(f'Mean Absolute Percentage Error (MAPE): {mape_score:.4f}%')
```

### Model Interpretation

What does this result mean? Is this a good or bad result?

```python
# Feature importance
feature_importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
# visualise the most important features
feature_importances.head(25).plot(kind='bar', figsize=(12,6))
```

![[Pasted image 20241014142753.png]]
```python
# Exporting the tree from the Random Forest model
tree = model.estimators_[0]
dot_data = export_graphviz(tree, out_file=None, 
                           feature_names=X.columns, 
                           filled=True, rounded=True, 
                           special_characters=True,
                           max_depth=4)
graph = graphviz.Source(dot_data)
graph.render("decision_tree", format="pdf")

# Displaying the decision tree
print('decision tree saved')
```

### Remodeling Data

```python
# Define the current date
current_date = datetime.now()

# Convert 'date' columns to datetime
df['date'] = pd.to_datetime(df['date'], dayfirst=True)

# calculate months
df['time_since_date'] = ((datetime.now() - df['date']).dt.days/30)

# Preparing the data for modeling
X = df.drop(columns=['price', 'date', 'address'])
y = df['price']

# Encoding categorical variables
X = pd.get_dummies(X, drop_first=True)

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating and training the Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Making predictions
y_pred = model.predict(X_test)

# checking MAPE
mape_score = mean_absolute_percentage_error(y_test, y_pred)
print(f'Mean Absolute Percentage Error (MAPE): {mape_score:.4f}%')

# Feature importance
feature_importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)

# visualise the most important features
feature_importances.head(25).plot(kind='bar', figsize=(12,6))
```
![[Pasted image 20241014143042.png]]
### Imputing Missing values 
#### (buildingarea)
```python
# creating a copy of the df 
df_impute = df.copy()

# check missing values
df.isnull().sum().buildingarea


# Step 1: Calculate the average BuildingArea grouped by Rooms, Type, and Bathroom
avg_building_area = df.groupby(['rooms', 'type', 'bathroom'])['buildingarea'].transform('mean')

# Step 2: Impute missing values in BuildingArea
df['buildingarea'] = df['buildingarea'].fillna(avg_building_area)

# check missing values
df.isnull().sum().buildingarea

# Preparing the data for modeling
X = df.drop(columns=['price', 'date', 'address'])
y = df['price']

# Encoding categorical variables
X = pd.get_dummies(X, drop_first=True)

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating and training the Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Making predictions
y_pred = model.predict(X_test)

# checking MAPE
mape_score = mean_absolute_percentage_error(y_test, y_pred)
print(f'Mean Absolute Percentage Error (MAPE): {mape_score:.4f}%')


# Feature importance
feature_importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
# visualise the most important features
feature_importances.head(25).plot(kind='bar', figsize=(12,6))
```
![[Pasted image 20241014143740.png]]
#### (yearbuilt)
```python
# check missing values
df_impute.isnull().sum().yearbuilt


# Step 1: Calculate the average YearBuilt grouped by Rooms, Type, and Bathroom
avg_year_built = df_impute.groupby(['type'])['yearbuilt'].transform('mean')

# Step 2: Impute missing values in YearBuilt
df_impute['yearbuilt'] = df_impute['yearbuilt'].fillna(avg_year_built)

# check missing values
df_impute.isnull().sum().yearbuilt


# Preparing the data for modeling
X = df_impute.drop(columns=['price', 'date', 'address'])
y = df_impute['price']

# Encoding categorical variables
X = pd.get_dummies(X, drop_first=True)

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating and training the Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Making predictions
y_pred = model.predict(X_test)

# checking MAPE
mape_score_impute = mean_absolute_percentage_error(y_test, y_pred)
print(f'Mean Absolute Percentage Error (MAPE): {mape_score_impute:.4f}%')

# MAPE Increased

# Feature importance
feature_importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
# visualise the most important features
feature_importances.head(25).plot(kind='bar', figsize=(12,6))
```

![[Pasted image 20241014143950.png]]

## Adding Additional Data
```python
# loading the data
salary = pd.read_csv('salary.csv')

salary.columns = salary.columns.str.lower()

# exploring the data
salary.head()

# exploring the data
salary.info()

#merging data
df = df.merge(salary,left_on='suburb',right_on='suburb',how='left')


# Preparing the data for modeling
X = df.drop(columns=['price', 'date', 'address'])
y = df['price']

# Encoding categorical variables
X = pd.get_dummies(X, drop_first=True)

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating and training the Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Making predictions
y_pred = model.predict(X_test)


# checking MAPE
mape_score = mean_absolute_percentage_error(y_test, y_pred)
print(f'Mean Absolute Percentage Error (MAPE): {mape_score:.4f}%')


# Feature importance
feature_importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)

# visualise the most important features
feature_importances.head(25).plot(kind='bar', figsize=(12,6))
```
![[Pasted image 20241014144309.png]]

# Visualizing Decision Tree


```python
# Exporting the tree from the Random Forest model
tree = rf_model.estimators_[99]
dot_data = export_graphviz(tree, out_file=None, 
                           feature_names=X.columns, 
                           filled=True, rounded=True, 
                           special_characters=True,
                           max_depth=4)
graph = graphviz.Source(dot_data)
graph.render("decision_tree", format="pdf")

# Displaying the decision tree
print('decision tree saved')
```
# Clarifying Commands used ML Model Training

#  Python's Scikit-Learn

## .fit

### Internals of the `fit()` Method

When the `fit()` method is called, several internal processes occur:

1. ***Data Validation****: The method checks the input data for inconsistencies or missing values. Scikit-Learn provides utilities to handle these issues, but it’s essential to preprocess the data correctly.

2. ***Parameter Initialization****: The model’s parameters are initialized. For example, in linear regression, the coefficients and intercept are set to initial values.

3. ***Optimization Algorithm****: The model uses an optimization algorithm (like gradient descent) to iteratively adjust the parameters, minimizing the loss function.\

4. ***Convergence Check****: The algorithm checks for convergence. If the parameters no longer change significantly, the training stops.

**Explanation:**
The `fit()` method in Scikit-Learn is used to train a [machine learning](https://www.geeksforgeeks.org/ml-machine-learning/) model. Training a model involves feeding it with data so it can learn the underlying patterns. This method adjusts the parameters of the model based on the provided data.

**Syntax:**
The basic syntax for the `fit()` method is:

model.fit(X, y)

- `X`: The feature matrix, where each row represents a sample and each column represents a feature.
- `y`: The target vector, containing the labels or target values corresponding to the samples in `X`.

**Steps:
****Step 1: Import the necessary libraries****
```python
import numpy as np  
from sklearn.linear_model import LinearRegression
```
****Step 2: Create Sample Data****
```python
X = np.array([[1], [2], [3], [4], [5]])  
y = np.array([1.5, 3.1, 4.5, 6.2, 7.9])
```
****Step 3: Initialize the model****
```python
model = LinearRegression()
```

****Step 4: Train the model****
```python
model.fit(X, y)
```

****Step 5: Make Predictions****
```python
predictions = model.predict(X)
```
## .predict

**Explanation**:**Python predict() function** enables us to **predict the labels of the data values** on the basis of the trained model.

**Syntax**:
```python
model.predict(data)
```

## .feature_importances

**Explanation**:
Feature importances are provided by the fitted attribute `feature_importances_` and they are computed as the mean and standard deviation of accumulation of the impurity decrease within each tree.

```python
import time

import numpy as np

start_time = time.time()
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
elapsed_time = time.time() - start_time


import pandas as pd

forest_importances = pd.Series(importances, index=feature_names)

fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=std, ax=ax)
ax.set_title("Feature importances using MDI")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()
```
![[Pasted image 20241014151240.png]]

## pd.Series()

**Explanation**: Pandas Series is a one-dimensional labeled array capable of holding data of any type (integer, string, float, python objects, etc.).
```python
# import pandas as pd
import pandas as pd
 
# simple array
data = [1, 2, 3, 4]
 
ser = pd.Series(data)
print(ser)

# Output
0    1
1    2
2    3
3    4
dtype: int64
```

![[Pasted image 20241014151701.png]]





## .transform()

Pandas `**DataFrame.transform()**` function call func on self producing a DataFrame with transformed values and that has the same axis length as self.

**Syntax:** DataFrame.transform(func, axis=0, *args, **kwargs)

**Example**:
```python
# importing pandas as pd 
import pandas as pd 

# Creating the DataFrame 
df = pd.DataFrame({"A":[12, 4, 5, None, 1], 
				"B":[7, 2, 54, 3, None], 
				"C":[20, 16, 11, 3, 8], 
				"D":[14, 3, None, 2, 6]}) 

# Create the index 
index_ = ['Row_1', 'Row_2', 'Row_3', 'Row_4', 'Row_5'] 

# Set the index 
df.index = index_ 

# Print the DataFrame 
print(df) 

```
![[Pasted image 20241014152037.png]]

```python
# add 10 to each element of the dataframe 
result = df.transform(func = lambda x : x + 10) 

# Print the result 
print(result) 

```
![[Pasted image 20241014152114.png]]