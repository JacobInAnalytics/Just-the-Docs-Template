#python #Machinelearning
## Advanced imputation methods

#### **Using Machine Learning to Predict Missing Age Values**

This method involves training a regression model to predict missing age values based on other features. It uses a machine learning model to predict missing age values based on other features, potentially providing more accurate imputations.

## **Breaking down the steps**

### 1. **Data Preparation:**
- Create a copy of the original data.
    
- Convert categorical variables 'Sex' and 'Embarked' into numerical values using `pd.get_dummies()`.
    
- Separate the data into `train` (rows with non-missing 'Age') and `test` (rows with missing 'Age') datasets.

```python
# Prepare the data for training the model
data_ml = data.copy()
data_ml = pd.get_dummies(data_ml, columns=['Sex', 'Embarked'], drop_first=True)
train = data_ml.dropna(subset=['Age'])
test = data_ml[data_ml['Age'].isnull()]
```



### 2. **Feature Selection:**
- Exclude non-relevant columns ('Age', 'Name', 'Ticket', 'Cabin') from `train` and `test` datasets.
    
- `X_train`: Features from rows with known ages.
    
- `y_train`: Known ages.

```python
X_train = train.drop(['Age', 'Name', 'Ticket', 'Cabin'], axis=1)
y_train = train['Age']
X_test = test.drop(['Age', 'Name', 'Ticket', 'Cabin'], axis=1)
```
### 3 **Model Selection**
- Initialize a Random Forest Regressor (`RandomForestRegressor`) with 100 trees (`n_estimators=100`) and a fixed random seed (`random_state=42`).
    
- Train the model using `X_train` and `y_train`.

``` python
# Train the model
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
```

		
### 4. **Age Prediction:**
  - Predict the missing ages using the trained model on `X_test`.
    
- Replace the missing ages in the original dataset with the predicted values.

```python
# Predict missing age values
predicted_ages = rf.predict(X_test)
data_ml.loc[data_ml['Age'].isnull(), 'Age'] = predicted_ages

```
### 5. ** Visualization:**

- Plot the age distribution after imputation to visualize the result.


``` python
# Plotting the distribution of Age after using machine learning to predict missing values
plt.figure(figsize=(20, 20))
data_ml['Age'].plot(kind='hist', bins=30, color='teal')
plt.title('Distribution of Age After Predicting Missing Values with ML', fontsize=20)
plt.xlabel('Age', fontsize=15)
plt.ylabel('Frequency', fontsize=15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()```
**Explanation:**

- The Random Forest Regressor uses multiple decision trees to predict missing age values based on patterns learned from the other features.
    
- Each tree in the forest makes an independent prediction, and the average of all these predictions is used for the final imputation.
    

![](https://static.au.edusercontent.com/files/BrtoBl0xV9rlWV6q1kOjRzQX)

This method leverages the relationships between various features in the dataset to provide a more accurate estimate of the missing ages.

```python
# Plotting the distribution of Age after using machine learning to predict missing values
plt.figure(figsize=(20, 20))
data_ml['Age'].plot(kind='hist', bins=30, color='teal')
plt.title('Distribution of Age After Predicting Missing Values with ML', fontsize=20)
plt.xlabel('Age', fontsize=15)
plt.ylabel('Frequency', fontsize=15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()
```
![[Pasted image 20240905220540.png]]

## Dealing with Null values 
### Method 1: Removing the Null Ages from the Dataset

**Explanation:** This method involves removing all rows that contain null values in the "Age" column. This may result in loss of data but ensures that only complete records are used in the analysis.

```python
# importing library
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# uploading the data
data = pd.read_csv('train.csv')

# creating a copy of the data
data_copy = data.copy()

# Removing rows with null ages
data_removed_nulls = data.dropna(subset=['Age'])

# Plotting the distribution of Age after removing nulls
plt.figure(figsize=(20, 20))
plt.hist(data_removed_nulls['Age'], bins=20, color='skyblue', edgecolor='black')
plt.title('Age Distribution After Removing Null Ages', fontsize=20)
plt.xlabel('Age', fontsize=15)
plt.ylabel('Frequency', fontsize=15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()
```
![[Pasted image 20240905215515.png]]

### Method 2: Populating Null Age Values with the Average Age

**Explanation:** This method replaces null values in the "Age" column with the average age of all passengers. This maintains the size of the dataset and provides a simple estimate for missing values.

```python
# Populating null Age values with the average age
average_age = data['Age'].mean()
data_filled_avg = data.copy()
data_filled_avg['Age'].fillna(average_age, inplace=True)

# Plotting the distribution of Age after filling with average age
plt.figure(figsize=(20, 20))
plt.hist(data_filled_avg['Age'], bins=20, color='lightgreen', edgecolor='black')
plt.title('Age Distribution After Filling Null Ages with Average Age', fontsize=20)
plt.xlabel('Age', fontsize=15)
plt.ylabel('Frequency', fontsize=15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()
```
![[Pasted image 20240905215635.png]]

### Method 3: Populating Null Age Values with the Average Age by Sex

**Explanation:** This method replaces null values in the "Age" column with the average age of passengers grouped by sex. This provides a more contextual estimate for missing values.

```python
# Populating null Age values with the average age by sex, Pclass, and Survived
data_filled_group_avg = data.copy()
data_filled_group_avg['Age'] = data.groupby(['Sex'],group_keys=False)['Age'].apply(lambda x: x.fillna(x.mean()))

# Plotting the distribution of Age after filling with group averages
plt.figure(figsize=(20, 20))
plt.hist(data_filled_group_avg['Age'], bins=20, color='lightcoral', edgecolor='black')
plt.title('Age Distribution After Filling Null Ages with Group Averages', fontsize=20)
plt.xlabel('Age', fontsize=15)
plt.ylabel('Frequency', fontsize=15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()
```
![[Pasted image 20240905215759.png]]

### Method 4. Populating Null Age Values with Random Values within the Same Group

**Explaination:** This method involves filling missing age values with random values within the same group defined by sex, Pclass, and survived. This method maintains the distribution of ages within each group but introduces randomness.

```python
# Populating null Age values with random values within the same group
data_filled_random_age = data.copy()
for group in data_filled_random_age.groupby(['Sex', 'Pclass', 'Survived']):
    idxs = group[1].index
    data_filled_random_age.loc[idxs, 'Age'] = group[1]['Age'].transform(lambda x: x.fillna(np.random.choice(x.dropna())))

# Plotting the distribution of Age after filling null values with random values within the same group
plt.figure(figsize=(20, 20))
data_filled_random_age['Age'].plot(kind='hist',
                         bins=30, # what does this do?
                          color='purple')
plt.title('Distribution of Age After Filling Null Values with Random Values within Group', fontsize=20)
plt.xlabel('Age', fontsize=15)
plt.ylabel('Frequency', fontsize=15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()
```


![[Pasted image 20240905215908.png]]
### Method 5 Populating null Age values with the average using Groupby

```python
data['freetime_after_school'] = data.groupby(['sex','address','school_absences'],group_keys=False)['freetime_after_school'].apply(lambda x: x.fillna(x.mean()))
```

```python
# Plot before and after imputation
fig, axes = plt.subplots(1, 2, figsize=(6, 8), sharey=True)

sns.histplot(data_original['freetime_after_school'], ax=axes[0], kde=True,color='red').set_title('Before Imputation')

sns.histplot(data['freetime_after_school'], kde=True, ax=axes[1], color='green').set_title('After Imputation')

plt.show()
```

![[Pasted image 20241014130351.png]]
## [[Python Dictionary#^543235|String]] functions

Real-world data is messy, and a lot of it comes in text form. That's where [[Python Dictionary#^543235|String]] Functions in Python are imperative for data science. You will need these skills very frequently to help with:

- cleaning up data: Removing weird characters, making everything look nice and tidy.
    
- creating new features: Pulling cool insights out of text you already have.
    

Getting good at this stuff makes you a sharper Python coder overall. It's like learning to crawl before you walk – master this, and you're set up to tackle other more complex python functions.

### len()

**Explanation:**
The len() function in Python returns the number of characters in a [[Python Dictionary#^543235|string]]. It's a fundamental tool for [[Python Dictionary#^543235|string]]]] manipulation and data cleaning in pandas DataFrames.

```python
# Apply len() function
average_email_length = df['email_address'].str.len().mean()
```
### str.replace()

>[!info] Series.str.replace
>(**_str_**, **_repl_**, **_n**=-1_, **_case**=
>None_, **_flags**=0_, **_regex**=False_)

**repl**:
str or callable

Replacement [[Python Dictionary#^543235|string]] or a callable. The callable is passed the regex match object and must return a replacement [[Python Dictionary#^543235|string]] to be used. See [`re.sub()`](https://docs.python.org/3/library/re.html#re.sub "(in Python v3.12)").
Replace each occurrence of pattern/regex in the Series/Index.

**Explanation:**

The replace() function substitutes specified sub[[Python Dictionary#^543235|string]]s within [[Python Dictionary#^543235|string]]s. It's crucial for data cleaning and standardization in pandas DataFrames.

```python
# Using replace for email address
df['email'] = df['email'].str.replace('@outlook.com', '@live.com')

or

# removing comma
df['first-last-name-adj'] = df['first-last-name'].str.replace(',', ' ')
```


### str.count() 

**Explanation:**

The count() function tallies the occurrences of a sub[[Python Dictionary#^543235|string]] within a [[Python Dictionary#^543235|string]]. It's valuable for frequency analysis in text data.

```python

# count the number of people who have Lisa as a first name
df['first-last-name-adj'].str.count('Lisa ').sum()

or


df['first-last-name-adj']
	.str.count(' Rhodes').sum()
```
**Key Components:**

str.count('Lisa '):

- Counts how many times 'Lisa' appears in each column name.
- The .sum() function then counts the number of times Lisa appears
### str.split()

>[!info] Series.str.split
>(**_str**=None_, _*_,** ****_n**=-1_, 
>**_expand**=False_, **_regex**=None_)

**Explanation:**

Splitting text to columns is a powerful technique for extracting structured data from [[Python Dictionary#^543235|string]] fields. It's especially useful when dealing with combined data fields.

```python
# Split 'full_name' into 'first_name' and 'last_name'
df[['first_name', 'last_name']] = df['first-last-name-adj'].str.split(' ', expand=True)


#Split the 'email' column into 'username' and 'domain' parts.
df[['username', 'domain']] = df['email_address'].str.split('@', expand=True)
```
**pat**:
str or compiled regex, optional

String or regular expression to split on. If not specified, split on whitespace.

**n**:
int, default -1 (all)

Limit number of splits in output. `None`, 0 and -1 will be interpreted as return all splits.

**expand**:
bool, default False

Expand the split [[Python Dictionary#^543235|string]]s into separate columns.

- If `True`, return DataFrame/MultiIndex expanding dimensionality.
    
- If `False`, return Series/Index, containing lists of [[Python Dictionary#^543235|string]]s.


**Key Components:**

str.split(' ', expand=True):

- Splits the 'full_name' [[Python Dictionary#^543235|string]] at the space character.
- expand=True creates new columns for each split component.
- The result is directly assigned to new columns 'first_name' and 'last_name'.
