#Regression #python
# Basic date functions in python: datetime library

## Use Cases for the `datetime` Library


- **Simple Date and Time Manipulations**:
    
    - **Lightweight**: The `datetime` library is part of Python’s standard library and is ideal for simple date and time manipulations without the overhead of a larger library.
        
    - **Example**: Calculating the number of days between two dates.


- **Standalone Applications**:
    
    - **Integration**: `datetime` is sufficient for applications that require basic date and time functionality without the need for data manipulation and analysis tools provided by pandas.
        
    - **Example**: Creating a countdown timer or logging the current time of an event.




- **Compatibility**:
    
    - **Interoperability**: The `datetime` module can be used in environments where installing third-party libraries like pandas is not feasible or necessary.
        
    - **Example**: Writing scripts for systems with strict library installation policies.




- **System-Level Operations**:
    
    - **Direct Use**: For system-level date and time operations, such as scheduling tasks or interacting with operating system time settings, the `datetime` module is typically sufficient.
        
    - **Example**: Scheduling a task to run at a specific time using the operating system’s task scheduler.


### Datetime.date

The object of the Date class represents the naive date containing year, month, and date according to the current Gregorian calendar. This date can be extended indefinitely in both directions. The January 1 of year 1 is called day 1 and January 2 or year 2 is called day 2 and so on.

>[!info]  **Syntax**: 
>class datetime.date(year, month, day)

```python
# datetime.date: Represents a date (year, month, day) without time.

from datetime import date
d = date(2024, 6, 30) 
d

#output: datetime.date(2024, 6, 30)

# checking what datatype "d" is
type(d)

# printing d
print(d) 

```
### Datetime.time

The time class creates the time object which represents local time, independent of any day.


>[!info] **Constructor Syntax:**
class datetime.time(hour=0, minute=0, second=0, microsecond=0, tzinfo=None, *, fold=0)

Returns tzinfo.tzname() is tzinfo is not None

```python

# datetime.time: Represents a time (hour, minute, second, microsecond) without date.
from datetime import time
t = time(14, 30, 0)
t


# datetime.datetime: Combines date and time into a single object.
from datetime import datetime
dt = datetime(2024, 6, 30, 14, 30, 0)
dt

now = datetime.now()
print(now)


```

### str Time
The ****Strftime()**** function is used to convert date and time objects to their [[Python Dictionary#^543235|string]] representation. It takes one or more inputs of formatted code and returns the [[Python Dictionary#^543235|string]] representation in Python.

>[!info] **Syntax:** 
datetime_obj.strftime(format)

**Parameters:**

- **Format:*** consists of various format codes that define specific parts of the date and time.

**Returns:*** It returns the [[Python Dictionary#^543235|string]] representation of the date or time object.


```python
#strftime(format): Converts the datetime to a [[Python Dictionary#^543235|string]] according to the given format.
dt.strftime('%Y-%m-%d %H:%M:%S')

#strptime(date_[[Python Dictionary#^543235|string]], format): Parses a [[Python Dictionary#^543235|string]] into a datetime object according to the given format.
dtp = datetime.strptime('2024-06-30 14:30:00', '%Y-%m-%d %H:%M:%S')
print(dtp)
```


![[Pasted image 20240907195831.png]]
### Adding and Subtracting dates/ timedelta()

For adding or subtracting Date, we use something called **timedelta()** function which can be found under the **DateTime** class. It is used to manipulate Date, and we can perform arithmetic operations on dates like adding or subtracting. **timedelta** is very easy and useful to implement.



```python

# Adding/Subtracting Days
from datetime import datetime, timedelta
now = datetime.now()

# adding dates
future_date = now + timedelta(days=10)

# Subtracting dates
past_date = now - timedelta(days=10)

# adding times 
future_time = now + timedelta(hours=5, minutes=30)

# subtracting times
past_time = now - timedelta(hours=3)
```


### Extracting Specific fields

```python
# reminder of what the now variable is saved as
now

# extracting year
year = now.year

# extracting month
month = now.month

# extracting day
day = now.day

# extracting hour
hour = now.hour

# extracting minute
minute = now.minute

# extracting second
second = now.second

```

# Basic date functions in Pandas

Use Cases for `Pandas` Library

- **Handling Large Datasets**:
    
    - **Efficiency**: Pandas is optimized for performance with large datasets, enabling fast and efficient operations.
        
    - **Example**: Analyzing a dataset with thousands or millions of rows of time series data.
        
- **Time Series Analysis**:
    
    - **Functionality**: Pandas offers powerful tools for time series analysis, including resampling, rolling windows, and time-based indexing.
        
    - **Example**: Calculating moving averages, resampling data to different frequencies, or aligning multiple time series.
        
- **DataFrame and Series Operations**:
    
    - **Integration**: Pandas seamlessly integrates date and time functionalities within its DataFrame and Series structures, making it easy to perform vectorized operations.
        
    - **Example**: Adding a new column to a DataFrame based on a date calculation or filtering rows based on date ranges.
        
- **Date and Time Parsing**:
    
    - **Convenience**: Pandas provides robust functions like `pd.to_datetime` for parsing date [[Python Dictionary#^543235|string]]s into datetime objects, handling various formats and common issues like missing data.
        
    - **Example**: Converting a column of date [[Python Dictionary#^543235|string]]s in a CSV file to datetime objects.
        
- **Complex Date Manipulations**:
    
    - **Advanced Features**: Pandas offers advanced features such as `DatetimeIndex` for efficient indexing and slicing of time series data.
        
    - **Example**: Selecting all data points within a specific month or generating a range of dates with a specific frequency.
        
- **Group By and Aggregation**:
    
    - **Functionality**: Pandas allows for grouping data by date or time and performing aggregations such as sum, mean, or count.
        
    - **Example**: Grouping sales data by month and calculating total sales per month.
        
- **Time Zone Handling**:
    
    - **Ease of Use**: Pandas provides straightforward methods for time zone conversions and adjustments.
        
    - **Example**: Converting a series of timestamps from one time zone to another.

```python
# import new library
import pandas as pd
```
### Timestamps()

>[!info]  **Syntax: **_class_ pandas.Timestamp
>(**_ts_input**=object object_, **_year**=None_, **_month**=None_, 
>**_day**=None_, **_hour**=None_, **_minute**=None_, **_second**=None_, **_microsecond**=None_)
>

Timestamp is the pandas equivalent of python’s Datetime and is interchangeable with it in most cases. It’s the type used for the entries that make up a DatetimeIndex, and other timeseries oriented data structures in pandas.


```python
# Timestamp: A pandas equivalent of Python’s datetime.datetime object, representing a single timestamp.
ts = pd.Timestamp('2023-06-30 14:30:00')
print(ts)

# now(): Returns the current local date and time as a Timestamp.
now = pd.Timestamp.now()
print(now)

# DatetimeIndex: A collection of Timestamps, similar to an index.
dti = pd.date_range(start='2023-06-01', end='2023-06-03', freq='D')
print(dti)
```

### to_pydatetime()

Return the data as an [[Python Dictionary#^d8476a|array]] of `datetime.datetime'objects.

```python
# DatetimeIndex: A collection of Timestamps, similar to an index.
dti = pd.date_range(start='2023-06-01', end='2023-06-03', freq='D')
print(dti)

# to_pydatetime(): Converts to a numpy array of Python datetime objects.
dti.to_pydatetime()


# to_datetime(): Converts a list or array of date-like [[Python Dictionary#^543235|string]]s to a DatetimeIndex.
dates = pd.to_datetime(['2023-06-30', '2023-06-08'])
print(dates)

# Period: Represents a single time span (e.g., month, quarter).
period = pd.Period('2023-06', freq='M')
print(period)

```

## Timedelta in pandas

> [!info] _class_ pandas.Timedelta
>(**_value**=object object _, **_unit**=None_, )

Represents a duration, the difference between two dates or times.

Timedelta is the pandas equivalent of python’s `datetime.timedelta` and is interchangeable with it in most cases.

>[!info] pandas.to_timedelta
(_arg_, _unit=None_, _errors='raise'_)

Timedeltas are absolute differences in times, expressed in difference units (e.g. days, hours, minutes, seconds). This method converts an argument from a recognized timedelta format / value into a Timedelta type.


```python
# Sample data with two dates
data = {'start_date': pd.to_datetime(['2024-06-01', '2023-06-15', '2023-07-01']),
        'end_date': pd.to_datetime(['2024-07-10', '2023-06-20', '2023-07-05'])}

df = pd.DataFrame(data)

# Adding and Subtracting Dates
df['future_date'] = df['start_date'] + pd.Timedelta(days=10)
df['past_date'] = df['start_date'] - pd.Timedelta(days=10)

# Calculate the timedelta between the two dates
df['timedelta_days'] = (df['end_date'] - df['start_date']).dt.days

# Extracting year, month, day, hour, minute, and second from 'start_date'
df['year'] = df['start_date'].dt.year
df['month'] = df['start_date'].dt.month
df['day'] = df['start_date'].dt.day
df['hour'] = df['start_date'].dt.hour
df['minute'] = df['start_date'].dt.minute
df['second'] = df['start_date'].dt.second

# checking the changes to the dateframe
df.head()
```
# Understanding Regression
### High-Level Explanation of What a Regression Model is Doing

A regression model is a type of predictive modeling technique used to understand the relationship between a dependent variable (often called the outcome or target) and one or more independent variables (also known as predictors or features). The primary goal of a regression model is to predict the value of the dependent variable based on the values of the independent variables. This is achieved by finding the best-fit line or curve that represents the relationship between the variables.

At a high-level sense, regression models work by:

- **Identifying Relationships**: They identify and quantify the relationships between the independent variables and the dependent variable. This relationship can be linear (straight line) or non-linear (curved line).
    
- **Fitting a Model**: They fit a mathematical model to the data, which minimizes the differences (errors) between the predicted values and the actual observed values of the dependent variable.
    
- **Making Predictions**: Once the model is trained, it can be used to make predictions on new, unseen data by applying the learned relationship to the new independent variables.
    
- **Evaluating Performance**: They provide metrics to evaluate the model's performance, such as how accurately it predicts the dependent variable and how well it generalizes to new data

![[Pasted image 20240907210117.png]]


## Key Steps to Prepare a Dataset Before Conducting a Regression Model

Before conducting a regression analysis, it is crucial to prepare the dataset properly to ensure the model's accuracy and reliability. Here are the key steps:

- **Data Collection**: Gather all relevant data that might influence the dependent variable. Ensure that the data is comprehensive and representative of the problem you are trying to solve.
    
- **Data Cleaning**:
    
    - **Handle Missing Values**: Identify and handle missing values by either removing them or imputing them with appropriate values (e.g., mean, median, mode, or using more sophisticated methods).
        
    - **Remove Duplicates**: Check for and remove duplicate records to avoid bias.
        
    - **Correct Errors**: Identify and correct any errors or inconsistencies in the data (e.g., incorrect data types, outliers).
        
- **Data Transformation**:
    
    - **Normalize or Standardize**: Scale the data to ensure that all features contribute equally to the model (e.g., using Min-Max scaling or Z-score standardization).
        
    - **Encode Categorical Variables**: Convert categorical variables into a numerical format that the model can understand, typically using techniques like one-hot encoding or label encoding.
        
- **Feature Selection**:
    
    - **Remove Irrelevant Features**: Exclude features that do not contribute to the model’s predictive power.
        
    - **Dimensionality Reduction**: Use techniques like Principal Component Analysis (PCA) to reduce the number of features while retaining the essential information.
        
- **Feature Engineering**:
    
    - **Create New Features**: Construct new features that might improve the model’s performance based on domain knowledge or exploratory data analysis.
        
    - **Transform Features**: Apply mathematical transformations to existing features to better capture the underlying patterns (e.g., log transformation, polynomial features).
        
- **Splitting the Data**:
    
    - **Train-Test Split**: Divide the dataset into training and testing sets to evaluate the model’s performance on unseen data. Typically, this is done using an 80-20 or 70-30 split.
        
    - **Cross-Validation**: Use cross-validation techniques to ensure the model’s robustness and to minimize overfitting.
        
- **Exploratory Data Analysis (EDA)**:
    
    - **Visualize Data**: Use plots and charts to understand the distributions, relationships, and potential outliers in the data.
        
    - **Summary Statistics**: Calculate summary statistics to gain insights into the data (e.g., mean, median, variance).
        
- **Assumption Checks**:
    
    - **Linearity**: Check if the relationship between the dependent and independent variables is linear.
        
    - **Independence**: Ensure that the observations are independent of each other.
        
    - **Homoscedasticity**: Verify that the residuals have constant variance.
        
    - **Normality**: Check if the residuals of the model are normally distributed.
        

Properly preparing the dataset through these steps helps in building a more accurate and reliable regression model, ultimately leading to better predictions and insights.


## .copy()

- A _shallow copy_ constructs a new compound object and then (to the extent possible) inserts _references_ into it to the objects found in the original.
Return a shallow copy of _x_.

## _pd.get_dummies_()

Convert categorical variable into dummy/indicator variables.

Each variable is converted in as many 0/1 variables as there are different values. Columns in the output are each named after a value; if the input is a DataFrame, the name of the original variable is prepended to the value.

>[!info] .get_dummies
>
>(**_data_**, **_prefix**=None_, **_prefix_sep**='_'_, **_dummy_na**=False_,
> **_columns**=None_, **_sparse**=False_, **_drop_first**=False_,** **_dtype=None_)

Example
```python

>>> pd.get_dummies(s)
       a      b      c
0   True  False  False
1  False   True  False
2  False  False   True
3   True  False  False


>>> pd.get_dummies(pd.Series(list('abc')), dtype=float)
     a    b    c
0  1.0  0.0  0.0
1  0.0  1.0  0.0
2  0.0  0.0  1.0

```
## .subplots()
Add a set of subplots to this figure.
>[!info] .subplots 
>(**_nrows**=1_, **_ncols**=1_, _*_, **_sharex**=False_, 
>**_sharey**=False_, **_squeeze**=True_, **_width_ratios**=None_, 
>**_height_ratios**=None_, **_subplot_kw**=None_, **_gridspec_kw**=None_)

**Parameters**:

**nrows, ncols**int, default: 1

Number of rows/columns of the subplot grid.

--- 
**sharex, sharey** bool or {'none', 'all', 'row', 'col'}, default: False

Controls sharing of x-axis (_sharex_) or y-axis (_sharey_):

- True or 'all': x- or y-axis will be shared among all subplots.
    
- False or 'none': each subplot x- or y-axis will be independent.
    
- 'row': each subplot row will share an x- or y-axis.
    
- 'col': each subplot column will share an x- or y-axis.
    
When subplots have a shared x-axis along a column, only the x tick labels of the bottom subplot are created. Similarly, when subplots have a shared y-axis along a row, only the y tick labels of the first column subplot are created. To later turn other subplots' ticklabels on, use [`tick_params`](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.tick_params.html#matplotlib.axes.Axes.tick_params "matplotlib.axes.Axes.tick_params").

When subplots have a shared axis that has units, calling [`Axis.set_units`](https://matplotlib.org/stable/api/_as_gen/matplotlib.axis.Axis.set_units.html#matplotlib.axis.Axis.set_units "matplotlib.axis.Axis.set_units") will update each axis with the new units.

Note that it is not possible to unshare axes.

--- 

**squeeze** bool, default: True

- If True, extra dimensions are squeezed out from the returned [[Python Dictionary#^d8476a|array]] of Axes:
    
    - if only one subplot is constructed (nrows=ncols=1), the resulting single Axes object is returned as a scalar.
        
    - for Nx1 or 1xM subplots, the returned object is a 1D numpy object [[Python Dictionary#^d8476a|array]] of Axes objects.
        
    - for NxM, subplots with N>1 and M>1 are returned as a 2D [[Python Dictionary#^d8476a|array]].
        
- If False, no squeezing at all is done: the returned Axes object is always a 2D [[Python Dictionary#^d8476a|array]] containing Axes instances, even if it ends up being 1x1.
--- 

**Example Below**
## Imputing the data

Imputation preserves all cases by **replacing missing data with an estimated value based on other available information**

``` python

# import library
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Create DataFrame
data = pd.read_csv('train.csv')

# Display the DataFrame
data.head()

# creating a backup copy of the data 
data_original = data.copy()

# Populating null Age values with the average age by sex, Pclass, and Survived
data['Age'] = data.groupby(['Sex', 'Pclass'],group_keys=False)['Age'].apply(lambda x: x.fillna(x.mean()))


# Plot before and after imputation
fig, axes = plt.subplots(1, 2, figsize=(6, 8), sharey=True)
sns.histplot(data_original['Age'], ax=axes[0], kde=True, color='red').set_title('Before Imputation')
sns.histplot(data['Age'], kde=True, ax=axes[1], color='green').set_title('After Imputation')
plt.show()

# Axes: Refers to an entire plot. As far as Matplotlib is concerned, the histogram we just made is an **axes object** — NOT a plot or graph! Axes is not the plural form of axis
```

![[Pasted image 20240907214101.png]]

### Explanation of Not Using The "Survived" Field

**Explanation:**

Using the Survived field for imputing the Age field can lead to data leakage. Data leakage occurs when information from outside the training dataset is used to create the model. This can lead to overly optimistic performance estimates and poor generalization to new data.

**Comments:**

Using the Survived field to impute Age can introduce bias because the survival status might be influenced by the age of the passengers, thereby distorting the model's understanding of the relationship between age and survival.

# Regression Model
**Explanation:**

A regression model, such as logistic regression, can be used to predict a binary outcome (like survival).

## **Step 1** Convert all categorical dimensions to numerical values.

**Why?**

Regression algorithms are based on mathematical operations and require numerical input. Categorical variables, which represent qualitative data, cannot be directly processed by these algorithms. Converting categorical variables into numerical formats allows the model to interpret and analyze relationships effectively.

```python
# Convert categorical variables into dummy/indicator variables
data = pd.get_dummies(data, columns=['Sex', 'Embarked'], drop_first=True)****
```

## **Step 2** Separate the data into features and target variables

**Why**

Separating the data into features (input variables) and target variables (output variable) clearly defines what the model needs to predict. Features provide the information used to make predictions, while the target variable is the outcome the model aims to predict.

```python
# Define features we will use for the model
X = data[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_male', 'Embarked_Q', 'Embarked_S']]

# define the target variable 
y = data['Survived']
```

## **Step 3** Separate the feature and target dimensions into train and test

**Why**

Separating data into training and testing sets allows us to validate the model's performance. Training the model on one subset and testing it on another helps assess how well the model generalizes to new, unseen data.

```python
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
```


## **Step 4** Training the logistic regression model

```python
# Train the logistic regression model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)
```

### A little bit of theory

**max_iter Parameter:**

This parameter sets the maximum number of iterations that the optimization algorithm can run to converge to the best solution. During each iteration, the algorithm updates the coefficients slightly, moving towards the direction that reduces the error.

If the algorithm converges (i.e., the changes in the coefficients become very small) before reaching the maximum number of iterations, it stops early. If the algorithm does not converge within the specified number of iterations, it stops and may not have found the best solution. This can happen if the data is complex or the learning rate is not well-tuned.

## **Step 5** Use the trained model to predict the output

```python
# Predict on the test set
y_pred = model.predict(X_test)
```

## **Step 6** Compare the results of the predicted output with the actual answers i.e. y_pred v y_test

```python
# Evaluate the model using accuracy and confusion matrix
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Confusion Matrix:\n{conf_matrix}')

#**Accuracy: 0.8555555555555555 Confusion Matrix: [[46 8] [ 5 31]]**
```


### Technical Reminder

**Confusion Matrix** a tool typically used to evaluate the performance of classification models, not regression models. It summarizes the number of correct and incorrect predictions made by the model, comparing actual target values with predicted values.

**Accuracy Score** a metric used to evaluate the correctness of predictions. For classification, it is the ratio of the number of correct predictions to the total number of predictions.

![[Pasted image 20240907220814.png]]

## **Step 7** Calculate feature importance
**Why**

Calculating feature importance helps in understanding which features have the most significant impact on the model's predictions. By identifying the most important features, we can keep the most relevant features and improve model performance.

```python
# Calculate feature importance
feature_importance = model.coef_[0]

# Create a DataFrame to display feature importance
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importance
}).sort_values(by='Importance', ascending=False)

# Plot feature importance
plt.figure(figsize=(6, 4))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title('Feature Importance')
plt.show()
```
![[Pasted image 20240907220903.png]]
### Understanding Feature Importance Scores

**Positive Importance Score:** A positive coefficient indicates that as the feature value increases, the likelihood of the positive class increases (assuming binary logistic regression). In other words, higher values of this feature are associated with a higher probability of the target being 1 (or the positive class).

**Negative Importance Score:** A negative coefficient indicates that as the feature value increases, the likelihood of the positive class decreases. This means that higher values of this feature are associated with a higher probability of the target being 0 (or the negative class).

## **Step 8** Transform the test data into the format required for the model

```python
# Import new test data
test_data = pd.read_csv('test.csv')

# Populating null Age values with the average age by sex, Pclass, and Survived
test_data['Age'] = test_data.groupby(['Sex', 'Pclass'],group_keys=False)['Age'].apply(lambda x: x.fillna(x.mean()))

# check for null values 
test_data.isnull().sum()

# PassengerId 0 Pclass 0 Name 0 Sex 
#0 Age 0 SibSp 0 Parch 0 Ticket 0 Fare 1 
#Cabin 327 Embarked 0 dtype: int64

# using an average of sex and PClass for the missing fare value
test_data['Fare'] = test_data.groupby(['Sex', 'Pclass'],group_keys=False)['Fare'].apply(lambda x: x.fillna(x.mean()))

# Preprocess the test data in the same way as the training data
test_data = pd.get_dummies(test_data, columns=['Sex', 'Embarked'], drop_first=True)

# Ensure the test data has the same columns as the training data
test_data = test_data.reindex(columns=X.columns, fill_value=0)

# Predict on the new test data
test_predictions = model.predict(test_data)

# adding the survived field back to the test data
test_data['Survived_predicated'] = test_predictions
```