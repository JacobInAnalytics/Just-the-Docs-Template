#python #keyfunctions #VisualisingData #RemovingNull
## [[Python Dictionary#^f58e5e|Lists]] vs [[Python Dictionary#^2fb0a5|Index]]

List is a [[Python Dictionary#^543235|string]]
Index is the order of the list

## Summary of the more complex python functions

### 1. [[Python Dictionary#^15aca8|Classes]]

**Explanation:**

- **What:** Classes are blueprints for creating objects (instances). They encapsulate data for the object and methods to manipulate that data.
    
- **How:** Defined using the `class` keyword, classes contain attributes (variables) and methods (functions) that define the behavior of the objects.
    
- **When Used:** Use classes when you want to model real-world entities or organize your code into reusable, modular components. They are fundamental in object-oriented programming for creating complex data structures and functionalities.
    

### 2. If-Else Functions

**Explanation:**

- **What:** If-else statements are conditional statements that execute different blocks of code based on certain conditions.
    
- **How:** Use the `if` keyword to specify a condition, followed by the block of code to execute if the condition is true. Optionally, use `elif` (else if) for additional conditions and `else` for the default block of code if none of the conditions are true.
    
- **When Used:** If-else statements are used when you need to perform different actions based on different conditions. They are fundamental in decision-making processes in programming.
    
```python
# Function to check if a number is positive, negative, or zero
#def = do command
def check_number(num): 
    if num > 0:
        return "Positive"
    elif num < 0:
        return "Negative"
    else:
        return "Zero"

check_number(0)

'Zero'

# Even odd checker
def check_even_off(num):
    if (num % 2==0):
        return "even"
    else:return "odd"

check_even_off(15)
'odd'
```

### 3. Loops

**Explanation:**

- **What:** Loops are used to execute a block of code repeatedly based on a condition or over a sequence.
    
- **How:** Python has two main types of loops: `for` loops (for iterating over a sequence like [[Python Dictionary#^f58e5e|lists]], [[Python Dictionary#^353183|tuples]], [[Python Dictionary#^543235|string]]) and `while` loops (which run as long as a condition is true).
    
- **When Used:** Use loops when you need to perform repetitive tasks, iterate over collections, or when the same block of code needs to be executed multiple times with different values.
    

```python
for i in range(1,6):
    print(i)

#Using the while function in a loop
num = 1
while num <= 5:
    print(num)
    num += 1  # Increment the number by 1 till 5 
```

#### Use of 'i' in Loops
**"i" is a temporary variable used to store the integer value** of the current position in the range of the for loop that only has scope within its for loop. You could use any other variable name in place of "i" such as "count" or "x" or "number". For instance in your for loop above, during the first iteration of the loop i = 1, then i=2 for the next iteration then, i= 3 and so on till 6.
### 4. Lambda Functions

**Explanation:**

- **What:** Lambda functions are small, anonymous functions defined with the `lambda` keyword. They are limited to a single expression.
    
- **How:** Lambda functions can take any number of arguments but only have one expression. They are often used as a quick, inline function.
    
- **When Used:** Use lambda functions for small, throwaway functions that are not reused elsewhere. They are commonly used with functions like `map()`, `filter()`, and `sorted()` where a simple function is needed for a short period.
    
```python

# lambda function to add two numbers
multiply = lambda x, y: x * y

# Test the lambda function
multiply(4, 2)

# Creating FareCategory using a lambda function

# The lambda function checks the fare amount and returns 'Low', 'Medium', or 'High' based on the specified conditions
data['FareCategory'] = data['Fare'].apply(lambda x: 'Low' if x < 20 else 
										  ('Medium' if x <= 50 else 'High'))

# Display the updated DataFrame to verify the changes
data[['PassengerId', 'Fare', 'FareCategory']].head(10)
```
### 5. Error Handling

**Explanation:**

- **What:** Error handling is the process of responding to and managing errors or exceptions that occur during program execution.
    
- **How:** Python uses `try`, `except`, `else`, and `finally` blocks to handle exceptions. Code that might throw an error is placed inside a `try` block, and `except` blocks are used to catch and handle specific exceptions.
    
- **When Used:** Error handling is used to manage and respond to runtime errors gracefully, without crashing the program. It is essential for building robust and user-friendly applications, especially when dealing with operations that can fail, such as file I/O, network requests, and user inputs.
```python
# Use Try and Except commands when error handling 
# Function to divide two numbers with error handling
def divide(a, b):
    try:
        result = a / b
    except ZeroDivisionError:
        return "Cannot divide by zero!"
    except TypeError:
        return "Invalid input type!"
    else:
        return result
```

### 6. Group by

>[!info]  Group by default function
>**DataFrame.groupby**  (**by**=None, **axis**=_NoDefault.no_default, **level**=None,
**as index**=True , **sort**=True, **group_keys**=True_, **observed**=No_Default.no_default, **dropna**=True)

**Group by Description**
Group DataFrame using a mapper or by a Series of columns.

A groupby operation involves some combination of splitting the object, applying a function, and combining the results. This can be used to group large amounts of data and compute operations on these groups.



**Parameters**

**by**: 
mapping, function, label, pd.Grouper or list of such

**Used to determine the groups for the groupby**. If `by` is a function, it’s called on each value of the object’s [[Python Dictionary#^2fb0a5|Index]]. If a dict or Series is passed, the Series or dict VALUES will be used to determine the groups (the Series’ values are first aligned; see `.align()` method). If a list or ndarray of length equal to the selected axis is passed (see the [groupby user guide](https://pandas.pydata.org/pandas-docs/stable/user_guide/groupby.html#splitting-an-object-into-groups)), the values are used as-is to determine the groups. A label or list of labels may be passed to group by the columns in `self`. Notice that a tuple is interpreted as a (single) key.

**Axis**: 
{0 or ‘index’, 1 or ‘columns’}, default 0

Split along rows (0) or columns (1). For Series this parameter is unused and defaults to 0.

**Think x or y axis**


**level**:
int, level name, or sequence of such, default None

If the axis is a MultiIndex (hierarchical), group by a particular level or levels. Do not specify both `by` and `level`.


**as index**:
bool, default True

Return object with group labels as the [[Python Dictionary#^2fb0a5|Index]]. Only relevant for DataFrame input. as_index=False is effectively “SQL-style” grouped output. This argument has no effect on filtrations (see the [filtrations in the user guide](https://pandas.pydata.org/docs/dev/user_guide/groupby.html#filtration)), such as `head()`, `tail()`, `nth()` and in transformations (see the [transformations in the user guide](https://pandas.pydata.org/docs/dev/user_guide/groupby.html#transformation)).


**sort**:

bool, default True

Sort group keys. Get better performance by turning this off. Note this does not influence the order of observations within each group. Groupby preserves the order of rows within each group. If False, the groups will appear in the same order as they did in the original DataFrame. This argument has no effect on filtrations (see the [filtrations in the user guide](https://pandas.pydata.org/docs/dev/user_guide/groupby.html#filtration)), such as `head()`, `tail()`, `nth()` and in transformations (see the [transformations in the user guide](https://pandas.pydata.org/docs/dev/user_guide/groupby.html#transformation)).

**group_keys**:
bool, default True

When calling apply and the `by` argument produces a like-indexed (i.e. [a transform](https://pandas.pydata.org/pandas-docs/stable/user_guide/groupby.html#groupby-transform)) result, add group keys to [[Python Dictionary#^2fb0a5|Index]] to identify pieces. By default group keys are not included when the result’s index (and column) labels match the inputs, and are included otherwise.

**observed**:
bool, default False

This only applies if any of the groupers are Categoricals. If True: only show observed values for categorical groupers. If False: show all values for categorical groupers.

**drop_na**: 
bool, default True

**If True, and if group keys contain NA value**s, NA values together with row/column will be dropped. If False, NA values will also be treated as the key in groups

**Example
```python
df_users['Age'] = df_users.groupby(['Gender','Country',
                    'Height','Weight'],
                    group_keys=False)['Age'].apply(
                        lambda x: x.fillna(x.mean()))
```


### 7.pd.cut 
cuts it into specific things

> [!info] pd.cut
>  (**_x_**, **_bins_**, **_right**=True_, **_labels**=None_, **_retbins**=False_
>  , **_precision**=3_, **_include_lowest**=False_, **_duplicates**='raise'_, **_ordered**=True_)

Bin values into discrete intervals.

Use cut when you need to segment and sort data values into bins. This function is also useful for going from a continuous variable to a categorical variable. For example, cut could convert ages to groups of age ranges. Supports binning into an equal number of bins, or a pre-specified [[Python Dictionary#^d8476a|array]] of bins.

**right**
bool, default True

Indicates whether bins includes the rightmost edge or not. If `right == True` (the default), then the bins `[1, 2, 3, 4]` indicate (1,2], (2,3], (3,4]. This argument is ignored when bins is an IntervalIndex.

**Example**

```python
# Define bin edges and labels
bin_edges = [0, 10, 20, 30, 40, 50, 60, float('inf')]
bin_labels = ['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60+']

#cut means cut up into this specific thing
# Use pd.cut() to categorize ages
dates_data['Age_buckets_pd_cut'] = pd.cut(dates_data['age_years'], bins=bin_edges, labels=bin_labels, right=False)

```

#### Square vs round brackets for rounding 
The notation may be a little confusing, but just remember that s**quare brackets mean the end point is included, and round parentheses mean it's excluded**. If both end points are included the interval is said to be closed, if they are both excluded it's said to be open

### 8. .apply 

> [!info] DataFrame.apply
> (_func_, _axis=0_, _raw=False_, _result_type=None_, _args=()_, _by_row='compat'_, _engine='python'_, _engine_kwargs=None_, _**kwargs_)

Apply a function along an axis of the DataFrame.

Objects passed to the function are Series objects whose index is either the DataFrame’s index (`axis=0`) or the DataFrame’s columns (`axis=1`). By default (`result_type=None`), the final return type is inferred from the return type of the applied function. Otherwise, it depends on the result_type argument

```python
dates_data['yearbucket'] = dates_data.age_years.apply(year_count)
```
## **Pros and Cons of Using Python for Data Manipulation and Visualization**

**Pros:**

- **Ease of Use:** Python's syntax is simple and readable, making it easy to learn and use.
    
- **Comprehensive Libraries:** Libraries like Pandas and Matplotlib provide extensive functionality for data manipulation and visualization.
    
- **Large Community:** A large community means plenty of resources, tutorials, and support for troubleshooting.
    
- **Integration Capabilities:** Python integrates well with other tools and technologies, making it versatile for various applications.
    
- **Open Source:** Python and its libraries are free and open-source, reducing the cost of implementation.
    

**Cons:**

- **Performance:** Python can be slower than some compiled languages like C++ or Java, particularly for very large datasets.
    
- **Memory Consumption:** Python can consume more memory, which might be an issue with limited resources.
    
- **Global Interpreter Lock (GIL):** The GIL can be a bottleneck in multi-threaded applications, limiting performance in CPU-bound tasks.
    
- **Error Handling:** Dynamic typing can lead to runtime errors that are sometimes harder to debug compared to statically typed languages.
    
- **Version Compatibility:** Occasionally, libraries or dependencies may have compatibility issues between different Python versions.

# Pandas: Merge, Union + Join

## Combining Dataframes

Just like with most things in python, there are lots of different ways to come to the same outcome.

To simplify things here, we're going to focus on the main functions I've used to combine datasets in my career.

When combining dataframes the first question to consider is:

> Which direction do you want to combine the data?

There's two directions you can use to combine dataframes:

1. Combine them to the **left or right** of each other (the leaflet analogy)

- When combining data to the left or right, you can use the **[pandas.merge](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.merge.html)** function
    

![](https://static.au.edusercontent.com/files/T2PvOggqyELDfphwaQTqkfLf)

2. Combine them on the **top or bottom** of each other (the torn piece of paper analogy)

- When combining data to the top or bottom, you can use the **[pandas.concat](https://pandas.pydata.org/docs/reference/api/pandas.concat.html)** function
    

![](https://static.au.edusercontent.com/files/gHitGcBzRp7bRZteyRBX4ghY)

### Checking if columns match

If data frames have the same column names and those column names are in the same order

You can check this programmatically using python what matches -- check out the code below
```python
df_users1.columns == df_users2.columns

#array([ True, True, True, True, True, True, True, True, True])
```

## Parameters 
- **Objs:** a list of dataframes to be concatenated
    
- **axis:** whether you join on the x or y axis of the dataframe
    
- **ignore_index:** When True, pandas ignores the existing index when joining data. This is useful when the original index isn't meaningful.
Remember to use **ignore_index = True** when the index isn't relevant to your existing data
### **Merge Operations:**

>[!info] DataFrame.merge
>(**Data_Frame**, **_how**='inner'_, **_on**=None_, **_left_on**=None_, **_right_on**=None_, 
>**_left_index**=False_, **_right_index**=False_, **_sort**=False_, **_suffixes**=('_x', '_y')_, 
>**_copy**=None_, **_indicator**=False_, **_validate**=None_)


Merge operations in Pandas combine DataFrames based on common columns or indices, allowing you to integrate data from different sources. [[Reference link](https://towardsdatascience.com/all-the-pandas-merge-you-should-know-for-combining-datasets-526b9ecaf184)]

![](https://static.au.edusercontent.com/files/5HDxJiP6oGVc7BBQc0EgqLef)

```python
# Create sample DataFrames
df1 = pd.DataFrame({'A': ['A0', 'A1', 'A2'],
                    'B': ['B0', 'B1', 'B2'],
                    'C': ['C0', 'C1', 'C2']},
                    index=['K0', 'K1', 'K2'])

df2 = pd.DataFrame({'D': ['D0', 'D1', 'D2'],
                    'E': ['E0', 'E1', 'E2'],
                    'F': ['F0', 'F1', 'F2']},
                    index=['K0', 'K2', 'K3'])


df1.head()

# Merge combines DataFrames based on common columns or indices
merged_df = pd.merge(df1, df2, left_index=True, right_index=True, how='outer')

merged_df.head()

|    |A|B|C|D|E|F|
|---|---|---|---|---|---|---|
|K0  |A0|B0|C0|D0|E0|F0|
|K1  |A1|B1|C1|NaN|NaN|NaN|
|K2  |A2|B2|C2|D1|E1|F1|
|K3  |NaN|NaN|NaN|D2|E2|F2|


# Creating new 
df3 = pd.DataFrame({'key': ['K0', 'K1', 'K2'], 'A': ['A0', 'A1', 'A2']})
df4 = pd.DataFrame({'key': ['K0', 'K2', 'K3'], 'B': ['B0', 'B2', 'B3']})

# Merge df3 and df4 on the 'key' column
merged_exercise2 = pd.merge(df3, df4, on='key', how='outer')

df5 = pd.DataFrame({'A': ['K0', 'K1', 'K2'], 'B': ['B0', 'B1', 'B2'], 'C': ['C0', 'C1', 'C2']}, index=['A0', 'A1', 'A2'])

df6 = pd.DataFrame({'D': ['D0', 'D1', 'D2'], 'E': ['K0', 'K2', 'K3'], 'F': ['F0', 'F1', 'F2']}, index=['E0', 'E1', 'E2'])

# Merge df5 and df6 on the shared column
merged_exercise2 = pd.merge(df5, df6, left_on='A', 
                                right_on="E",
                                how='left')

merged_exercise2.head()
```
![[Pasted image 20240906103030.png]]
### **Union Operations:**

#### To join vertically 

Union operations (also known as concatenation) in Pandas vertically combine DataFrames, stacking them on top of each other to create a single, longer DataFrame. [[Reference link](https://medium.com/analytics-vidhya/a-tip-a-day-python-tip-5-pandas-concat-append-dev-skrol-18e4950cc8cc)]

![](https://static.au.edusercontent.com/files/qIt28ZtJEpXDJ964CyFQthgO)

>[!info] pandas.concat
>(**_objs_**, _*_, **_axis**=0_, **_join**='outer'_, **_ignore_index**=False_, **_keys**=None_,
> **_levels**=None_, **_names**=None_, **_verify_integrity**=False_, **_sort**=False_, **_copy**=None_)
>
```python
# Union (concatenation) combines DataFrames vertically
union_df = pd.concat([df1, df2])

# Union (concatenation) combines DataFrames horizontally
horiz_union_df1 = pd.concat([df1, df2], axis=1)

# Concatenate two DataFrames and reset the index.
union_exercise = pd.concat([df1, df2], ignore_index=True).reset_index()
union_exercise = union_exercise.rename(columns={'index': 'new_index'})
```

### **Join Operations:**

Join operations in Pandas combine DataFrames based on their indices, similar to SQL joins, allowing you to merge data from different sources using a common key. [[Reference Link](https://medium.com/@rslavanyageetha/joins-in-pandas-6d95a2ba8a74)]

![](https://static.au.edusercontent.com/files/CQX1DI5nZ9d9fYquzeKO92LW)

>[!info] DataFrame.join
>(**_other_**, **_on**=None_, **_how**='left'_, **_lsuffix**=''_, **_rsuffix**=''_, **_sort**=False_, **_validate**=None_

```python
# Join combines DataFrames based on their indices
joined_df = df1.join(df2, how='outer

					 
					 
# Answer for left join 
left_join_exercise = df1.join(df2, how='left')

					 
					 
# Answer for saving a csv
joined_df.to_csv('output_data.csv', index=False)
```