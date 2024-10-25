# Example Data Analysis Week 7

![[Pasted image 20241014140729.png]]

**Importing Packages
```python 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np


# Load the data
df = pd.read_csv('train.csv')
```

## Initial Data Exploration

Let's start by looking at the first few rows of our dataset and getting some basic information about it.

```python
# Display the first few rows
df.head()

# Get information about the dataset
df.info()

# Display the shape of the dataset
df.shape
```
### Handling Missing Values

Now, let's check for and handle any missing values in our dataset.

```python
# Check for missing values
df.isnull().sum()

# For this example,group by sex and address
columns_to_impute = ['traveltime', 'freetime_after_school']

for column in columns_to_impute:
    df[column] = df.groupby(['sex', 'address'])[column].transform(lambda x: x.fillna(x.mean()))

# Check again to make sure all missing values are handled
df.isnull().sum()
```

## Data Preprocessing

### Creating New Columns

We'll create some new columns that might add value to our analysis.

```python
# Create a new column for total education (mother + father)
df['total_education'] = df['mother_education'].map({'none': 0, 'primary education': 1, '5th to 9th grade': 2, 'secondary education': 3, 'higher education': 4}) + \df['father_education'].map({'none': 0, 'primary education': 1, '5th to 9th grade': 2, 'secondary education': 3, 'higher education': 4})
```
```python
# Create a binary column for whether both parents have jobs
df['both_parents_employed'] = ((df['Mothers_job'] != 'at_home') & (df['Fathers_job'] != 'at_home')).astype(int)


# Create a column for total study time (travel time + study time)
df['total_study_time'] = df['traveltime'] + df['studytime']
```
## Data Analysis and Visualization

### Pivot Table Analysis

Let's create a pivot table to analyze the relationship between various factors and the school grade percentage.

```python
# Create a pivot table exploring some of the data
pivot_one = pd.pivot_table(df, values='school_grade_percentage', 
                       index=['sex', 'address'], 
                       columns=['studytime'], 
                       aggfunc='mean')
```

```python
# Visualize the pivot table
plt.figure(figsize=(12, 6))
sns.heatmap(pivot_one, annot=True, cmap='YlGnBu')
plt.title('Average School Grade Percentage by Sex, Address, and Study Time')
plt.show()
```
![[Pasted image 20241014132247.png]]
### Correlation Analysis

We'll analyze the correlation between numerical variables.

```python
# Select numerical columns
numerical_columns = df.select_dtypes(include=[np.number]).columns

# Compute correlation matrix
correlation_matrix = df[numerical_columns].corr()

# Visualize correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix of Numerical Variables')
plt.show()
```
![[Pasted image 20241014132330.png]]

### Distribution of School Grade Percentage

Let's visualize the distribution of our target variable.
```python
plt.figure(figsize=(10, 6))
sns.histplot(df['school_grade_percentage'], kde=True)
plt.title('Distribution of School Grade Percentage')
plt.xlabel('School Grade Percentage')
plt.ylabel('Count')
plt.show()
```
![[Pasted image 20241014132451.png]]

## Predictive Modeling

Now, we'll create a regression model to predict the school grade percentage.

```python
# Select numerical columns
numerical_columns = df.select_dtypes(include=[np.number]).columns

# Select features for the model
features = ['age', 'studytime', 'failures', 'school_absences', 'total_education', 'both_parents_employed', 'total_study_time']

X = df[features]
y = df['school_grade_percentage']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
### Model Training
```python
# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate MSE and R-squared
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared Score: {r2}")

# Visualize actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted School Grade Percentage')
plt.show()
```

![[Pasted image 20241014132733.png]]
### Feature Importance

Finally, let's look at the importance of each feature in our model.
```python
# Get feature importances
importances = pd.DataFrame({'feature': features, 'importance': abs(model.coef_)})
importances = importances.sort_values('importance', ascending=False)


# Visualize feature importances
plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=importances)
plt.title('Feature Importances')
plt.xlabel('Absolute Coefficient Value')
plt.ylabel('Feature')
plt.show()
```

![[Pasted image 20241014132824.png]]
## Question?

What's missing from this model?
Let's try that again

```python
# Load the data
df_again = pd.read_csv('train.csv')

# Create a new column for total education (mother + father)
df['total_education'] = df['mother_education'].map({'none': 0, 'primary education': 1, '5th to 9th grade': 2, 'secondary education': 3, 'higher education': 4}) + \ df['father_education'].map({'none': 0, 'primary education': 1, '5th to 9th grade': 2, 'secondary education': 3, 'higher education': 4})

# Create a binary column for whether both parents have jobs
df['both_parents_employed'] = ((df['Mothers_job'] != 'at_home') & (df['Fathers_job'] != 'at_home')).astype(int)

# Create a column for total study time (travel time + study time)
df['total_study_time'] = df['traveltime'] + df['studytime']

# check null values
df_again.isnull().sum()

# If there are missing values, we'll impute them using the mean of the group
# For this example, we'll group by sex and address
columns_to_impute = ['traveltime', 'freetime_after_school']

for column in columns_to_impute:
    df_again[column] = df_again.groupby(['sex', 'address'])[column].transform(lambda x: x.fillna(x.mean()))

# checking null values again
df_again.isnull().sum()

# Select categorical columns
categorical_columns = ['sex', 'address', 'mother_education', 'father_education', 'Mothers_job', 'Fathers_job','paid_tutorials', 'extra_curricular_activities', 'wants_higher_education','home_internet_access', 'romantic_relationship']

# Create dummy variables
df_again_encoded = pd.get_dummies(df_again, columns=categorical_columns, drop_first=True)

# Select features for the model (all columns except the target)
X = df_again_encoded.drop('school_grade_percentage', axis=1)
y = df_again_encoded['school_grade_percentage']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
linear_model = LinearRegression()

# Train models
linear_model.fit(X_train, y_train)

# Make predictions
linear_pred = linear_model.predict(X_test)

# Evaluate models
def evaluate_model(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f"{model_name} Performance:")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"Root Mean Squared Error: {rmse:.2f}")
    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"R-squared Score: {r2:.2f}\n")

# checking the results of the model 
evaluate_model(y_test, linear_pred, "Linear Regression")

```
## Model Performance Commentary

The evaluation metrics we've chosen (MSE, RMSE, MAE, and R-squared) provide a comprehensive view of our models' performance:

1. Mean Squared Error (MSE): This metric penalizes larger errors more heavily. It's useful for understanding the overall magnitude of the error, but it's sensitive to outliers.
    
2. Root Mean Squared Error (RMSE): This is the square root of MSE. It's in the same unit as our target variable, making it more interpretable. It represents the standard deviation of the residuals.
    
3. Mean Absolute Error (MAE): This metric represents the average absolute difference between predicted and actual values. It's less sensitive to outliers compared to MSE and RMSE.
    
4. R-squared: This metric represents the proportion of variance in the dependent variable that's predictable from the independent variable(s). It ranges from 0 to 1, with 1 indicating perfect prediction.
```python
plt.figure(figsize=(5, 5))
plt.scatter(y_test, linear_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Linear Regression: Actual vs Predicted')

plt.tight_layout()
plt.show()
```
![[Pasted image 20241014133435.png]]
**THIS MODEL IS OVERFIT

# Week 7 Being Impactful with data
5 Lessons about being impactful about data.
- Data! = Decisions (Data does not equal decisions)
- Alone == Danger Zone ()
- Leverage > Loops
- Potential == Pretentious 
- Lim F(you) = Max (decency) 

Define Impact, Making these better of faster:
- Focusing attention (Analytics)
- Making decisions (Statistics)
- Automating tasks  (ML/AI)


Questions to ask
	"Imagine we've done all the hard work and 
	developed a perfection solution. what will you do with it?"

## Data! = Decisions 
- get clear on decision makers
- clarify default and alternate decisions
- understand what it will take
- constantly sync and reassess
- communicate and advocate for what matters
- get communication training.

## Alone == Danger Zone ()

- Steaks (Not having committed stakeholders)
>- You do some work 
>- Then go out to try and convince others to use it
>- no one cares, has the time, etc

Creating a bus factor
>1. you do some work that's repeated/ maintained 
>2. No one else can do it
>3. You stop and does your impact

Summary
Steaks
- Catch and stop solo work
- Don't start projects without committed executors (in writing ideally)

Buses
- Identify & call out bus factor situations
- invest time with data peers and mentors

## Leverage > Loops
Try to improve metrics that everyone uses.

Work more big picture.

small problems = small impact. 
Impact goes both ways small impact = small problems.

Find (or create) Leverage so the same work = 10X the impact. 

Summary
- Identify your loops
- look for repeated problems/ work
- look for shared usage / influence 
- prioritise gaining leverage
- require degrees of leverage (priorities)
- think big

## Potential == Pretentious 
"potential" == ...
- Time spent
- No impact yet
- no promises
- more (unknown) work 
- Risks (eg lessons #5)

Impact == Impact

Summary
 - Get super clear on the desired impact
 - constantly drive towards it!
 - Constantly evaluate risks
 - Fight sunk costs, be okay to pivot fast
 - (managers) Communicate to team 

## Lim F(you) = Max (decency) 
Question:
Where does the data come from and how is it managed? 

Level 1
Quality, Accessible **Data**

Level 2 
easy-to-use **Tools**

Level 3 
Basic **Skills**

Level 4 
Data **Culture**


Summary
- Audit your (prospective) workplace on these levels


## The Importance of Learning Python for Data Science Through Hands-On Experimentation

While foundational knowledge is critical, the path to becoming a proficient data scientist is paved with experimentation and hands-on practice. Python’s user-friendly nature and its powerful data science libraries make it an ideal language for this purpose. By experimenting and playing with data yourself, you not only gain practical skills but also cultivate a deeper understanding and appreciation for the art and science of data analysis. So dive in, explore datasets, and let your curiosity guide your learning journey, knowing that perseverance through challenges will lead to success.

#### **Overcoming the Fear of Failure**

One of the most challenging aspects of learning data science is dealing with the inevitable failures and setbacks. Encountering errors, facing data that doesn't make sense, or having models that don't perform well can be frustrating and disheartening. However, these moments of failure are crucial learning experiences. Each mistake teaches you something new and helps you refine your skills. It's important to push through these difficult times, as persistence and resilience are key to mastering data science. Embrace the failures as part of the learning journey, knowing that each hurdle overcome is a step closer to proficiency.

#### **Develop Practical Skills**

Experimenting with Python allows you to apply theoretical concepts to real-world problems, solidifying your understanding. By working on actual datasets, you learn to clean, manipulate, and analyze data, gaining practical skills that are directly transferable to your professional work.

#### **Enhance Problem-Solving Abilities**

Data science often involves tackling complex and ambiguous problems. Through experimentation, you develop critical problem-solving skills. By facing and overcoming real challenges, such as missing data, noisy data, or computational inefficiencies, you become adept at devising creative solutions.

#### **Foster Creativity and Innovation**

Playing with data encourages a sense of curiosity and innovation. When you explore datasets on your own, you can test hypotheses, discover patterns, and derive insights that structured exercises might not cover. This exploratory mindset is crucial for making groundbreaking discoveries and driving innovation in your field.

#### **Build Confidence**

Confidence comes with practice. By continuously working on diverse datasets and projects, you build confidence in your abilities to handle different types of data and various analytical tasks. This self-assurance is vital when taking on new projects or transitioning to more advanced topics in data science.

#### **Learn to Use Essential Tools and Libraries**

Python offers a rich set of libraries such as pandas, NumPy, Matplotlib, and Scikit-learn, which are indispensable for data science. Experimenting with these tools helps you understand their capabilities and limitations, enabling you to use them effectively in your analyses. Hands-on practice ensures that you are not just familiar with these tools but proficient in their application.

#### **Prepare for Real-World Applications**

The ultimate goal of learning data science is to apply it in real-world scenarios. By experimenting with actual datasets, you simulate real-world conditions, preparing you for the challenges you will face in professional settings. This experience is invaluable for making informed decisions based on data.

#### **Stay Engaged and Motivated**

Learning by doing keeps you engaged and motivated. The satisfaction of solving a problem or uncovering a significant insight from your data can be incredibly rewarding. This intrinsic motivation drives continuous learning and improvement, which is essential in a field as dynamic as data science.

