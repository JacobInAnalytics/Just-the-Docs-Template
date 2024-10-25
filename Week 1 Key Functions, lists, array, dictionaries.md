# Tags: 
#keyfunctions #python

## F-[[Python Dictionary#^543235|string]]s

The 'f' before the [[Python Dictionary#^543235|string]] in the print function indicates that this is an f-[[Python Dictionary#^543235|string]], or formatted [[Python Dictionary#^543235|string]] literal, in Python. Here's a breakdown of why it's used:

- F-[[Python Dictionary#^543235|string]]s were introduced in Python 3.6 to make [[Python Dictionary#^543235|string]] formatting more convenient and readable.
- The 'f' prefix allows you to embed expressions inside [[Python Dictionary#^543235|string]] literals, enclosed in curly braces {}.
- In this case, {name} inside the [[Python Dictionary#^543235|string]] will be replaced with the actual value of the 'name' variable when the function is called.
- Without the 'f', it would be a regular string, and {name} would be treated as literal text rather than a placeholder for the variable's value.

## Python libraries 
**Introduction:** In Python, libraries (or modules) are collections of pre-written code that you can use to perform common tasks without having to write the code from scratch. Importing a library allows you to access its functions, classes, and variables in your own code.

**How to Import Libraries:** There are several ways to import libraries in Python. Here are the most common methods:

**Basic Import:**

```python
import library_name
```

This statement imports the entire library, and you can use its functions and classes by prefixing them with the library name.

**Example:**

```python
import math print(math.sqrt(16)) # Output: 4.0`
```


**Import Specific Functions or Classes:**
```python
from library_name import specific_function_or_class
```

This imports only the specified function or class from the library, making it directly accessible without the library prefix.

**Example:**
```python
from math import sqrt print(sqrt(16)) # Output: 4.0
```

**Import with Alias (Using "as"):**

```python
import library_name as alias
```

This imports the entire library but allows you to use a shorter or more convenient alias to reference it. The `as` keyword is used to create the alias.

**Example:**

```python
import pandas as pd df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]}) print(df)
```


**Import Specific Functions or Classes with Alias:**

```python
from library_name import specific_function_or_class as alias
```

This allows you to import a specific function or class and give it a shorter or more convenient alias.

**Example:**

```python
from math import sqrt as square_root print(square_root(16)) # Output: 4.0
```

## Lists
- **What:** [[#Lists]] are ordered collections of items (elements) which can be of different types (integers, [[Python Dictionary#^543235|string]], objects, etc.).
    
- **How:** Defined using square brackets `[]` and elements are separated by commas.
    
- **Usage:** Commonly used to store and manipulate a collection of related items.

**Practice Example**
```python
vegetables = ["carrot", "broccoli", "spinach", "potato", "pepper"]
vegetables.append("tomato")
second_vegetable = vegetables[1]
print(vegetables)  # Output: ['carrot', 'broccoli', 'spinach', 'potato', 'pepper', 'tomato']
print(second_vegetable)  # Output: broccoli

```


## Arrays
- **What:** [[#Arrays]] are similar to lists but are more efficient for numerical operations. They require the `numpy` library.
    
- **How:** Defined using `numpy.array()`.
    
- **Usage:** Used for performing mathematical and logical operations on large collections of data efficiently.

**Practice Examples**
```python
import numpy as np

numbers = np.array([2, 4, 6, 8, 10, 12])
multiplied_numbers = numbers * 3
print(multiplied_numbers)
```

## Ranges
- **What:** [[#Ranges]] generate a sequence of numbers. They are commonly used in loops.
    
- **How:** Defined using the `range()` function.
    
- **Usage:** Used to create a sequence of numbers for iteration.

```python
number_range = range(10, 21)
number_list = list(number_range)
print(number_list) 
```

## Dictionaries
- **What:** [[#Dictionaries]] are collections of key-value pairs. Each key is unique and maps to a value.
    
- **How:** Defined using curly braces `{}` with keys and values separated by a colon `:`.
    
- **Usage:** Used to store and retrieve data using keys.

```python
book = {"title": "1984", "author": "George Orwell", "year": 1949}
book["genre"] = "Dystopian"
print(book)
```
## Index starts at 0

- **What:** Python uses zero-based indexing, meaning the first element of a sequence is accessed with index 0.
 
- **How:** Lists, [[Python Dictionary#^543235|string]], and other sequence types use zero-based indexing.
 
- **Usage:** Important for accessing elements in sequences accurately.



## Python

**Introduction to Pandas:** Pandas is a powerful and versatile open-source data analysis and manipulation library for Python. It is widely used in data science and analytics because it provides easy-to-use data structures and data analysis tools. The name "Pandas" is derived from "panel data," a term for multidimensional structured data sets.

**Key Features of Pandas:**

- **Data Structures:**
    
    - **Series:** A one-dimensional [[#Arrays]]-like object that can hold any data type (integers, string, floats, etc.). Each element in a Series has an associated label called an index.
        
    - **DataFrame:** A two-dimensional, tabular data structure similar to a table in a database or an Excel spreadsheet. It consists of rows and columns, where each column can be a different data type.
        
- **Data Manipulation:**
    
    - **Reading Data:** Pandas can read data from various file formats, such as CSV, Excel, SQL databases, and more.
        
    - **Cleaning Data:** Provides tools for handling missing data, removing duplicates, and performing data transformations.
        
    - **Filtering and Selecting Data:** Allows you to select specific rows and columns based on conditions.
        
    - **Aggregating Data:** Supports group-by operations, which let you group data and apply aggregate functions like sum, mean, and count.
        
- **Data Analysis:**
    
    - **Descriptive Statistics:** Easily calculate statistics such as mean, median, standard deviation, and more.
        
    - **Time Series Analysis:** Offers robust support for working with time series data, including date range generation and frequency conversion.
        
    - **Merging and Joining:** Combine multiple DataFrames using different types of joins (inner, outer, left, right).
        

**Why Use Pandas?**

- **Ease of Use:** Pandas is designed to make data manipulation and analysis fast and easy. Its syntax is intuitive and its functions are powerful, allowing you to perform complex operations with just a few lines of code.
    
- **Efficiency:** Pandas is built on top of NumPy, another powerful library for numerical computing in Python. This makes Pandas highly efficient for large datasets.
    
- **Integration:** It integrates well with other data science libraries in Python, such as Matplotlib for plotting and Seaborn for statistical graphics.

## Pandas importing data

### Naming Conventions for Data Imports

Using consistent and descriptive naming conventions for your data imports helps improve the readability and maintainability of your code. Here are some tips:

1. **Use Descriptive Names:** Name your variables based on the content of the data. For example, use students_grades for a dataset containing student grades, or sales_data for sales information.
    
2. **Avoid Generic Names:** Avoid using generic names like data or df unless you are working with very short scripts or temporary variables.
    
3. **Follow Python Naming Conventions:** Use lowercase letters with underscores to separate words (snake_case). This is the standard naming convention in Python.
    

### Other data import functionality in pandas

The best way to find all the datatypes pandas will help support is honestly by Googling it. Let's try and find the relevant pandas documentation together.

**Python documentation**
https://pandas.pydata.org/docs/user_guide/index.html#user-guide

**Example of importing data types**

```Python
#import library
import pandas as pd

# this is how you import a csv 
'''
The pd.read_csv() function reads a CSV file into a Pandas DataFrame,
making it easy to manipulate and analyze the data.
'''
csv_data = pd.read_csv('train.csv')

# this is how you import a xlsx file
'''
The pd.read_excel() function reads an excel file with many tabs into a 
Pandas DataFrame, making it easy to manipulate and analyze the data.
'''
excel_data = pd.read_excel('titanic_data.xlsx')
```

## Python Key functions
```Python
# import pandas
import pandas as pd

# Load the dataset
titanic_data = pd.read_csv('train.csv')
```
### Select a column

There's a few ways to select a column in the dataframe using. This let's pandas and python know that that's the specific column you want to make changes to.

- Bracket Notation ([]): df['column_name']
- Dot Notation (.): df.column_name

```Python
titanic_data.Name

titanic_data.Pclass
```
### head()

**Explanation:** The head() function by default returns the first 5 rows of the DataFrame. You can pass a number to the function to display more or fewer rows e.g. titanic_data.head(10)

```Python
# Display the first 5 rows of the DataFrame
titanic_data.head(6)

titanic_data.head(11)
```
### Important note:

When writing code in python, not in jupyter notebook you have to use the following logic to print your output.

---

```
print(titanic_data.head())
```

---

Try this below to see what the output is

```python
print(titanic_data.head())
```

### tail()

**Explanation:** The tail() function by default returns the last 5 rows of the DataFrame. You can pass a number to the function to display more or fewer rows e.g. titanic_data.tail(10)

```python
titanic_data.tail()

titanic_data.tail(8)
```

### shape

**Explanation:** The shape attribute returns a [[Python Dictionary#^353183|tuple]] representing the dimensions of the DataFrame. The first element is the number of rows, and the second element is the number of columns.

```python
titanic_data.shape

titanic_data.shape() #This is an error ('tuple' object is not callable)
```

### info()

**Explanation:** The info() function displays the DataFrame’s structure, including the number of entries, column names, non-null counts, and data types._italic text_

```python
titanic_data.info() # Use without tuples

titanic_data.info
```

### describe()

**Explanation:** The describe() function returns summary statistics like count, mean, standard deviation, minimum, 25th percentile, median (50th percentile), 75th percentile, and maximum for numerical columns.

```python
titanic_data.describe()

#How would you modify describe() to include statistics for all columns, including non-numerical ones?

titanic_data.describe(include='all')
```

## .sort_values

**Explanation** Pandas sort_values() function sorts a data frame in Ascending or Descending order of passed Column. Used for signal parameter sorting. 

**Syntax****: DataFrame.sort_values(by, axis=0, ascending=True, inplace=False, kind=’quicksort’, na_position=’last’)

```python
# importing pandas package
import pandas as pd

# making data frame from csv file`

data = pd.read_csv("nba.csv")

# sorting data frame by name`

data.sort_values("Name", axis=0, ascending=True,inplace=True, na_position='last')

# display
data
```
**Output**
![[Pasted image 20241014145544.png]]
### value_counts()

**Explanation**: The value_counts() function is used to count the occurrences of unique values in a specified column.

```python
titanic_data.Embarked.value_counts()

#How would you find the unique values and their counts in the 'Pclass' column?

titanic_data.Pclass.value_counts()


# value counts sorted in ascending order 
df.Country.value_counts(ascending=False)
```

### NULL values

**Explanation:** The isnull() function checks for missing values and sum() aggregates the count of missing values for each column

```python
titanic_data.isnull().sum()

#How would you check if there are any rows with missing values in the 'Age' column?

titanic_data.Age.isnull().sum()
```

### sum()
The `sum()` function returns a number, the sum of all items in an iterable.

```python
titanic_data.Age.isnull().sum()
```

### nunique()
Count the number of distinct elements in specified axis.
```python
# use the describe function to learn how many unique first names are in the data

df["First Name"].nunique()
```

### fillna

**Explanation:** The fillna() function fills missing values with the specified value. The inplace=True parameter modifies the DataFrame directly.

```python
# fillna
titanic_data['Age'].fillna(titanic_data['Age'].mean())

# what does the inplace clause do?
titanic_data['Age'].fillna(titanic_data['Age'].mean(), inplace=True)

# inplace=True replaces the main dataframe 
# inplace=False creates a new dataframe that can be saved by assigning a varaible 


# How would you fill missing values in the 'Cabin' column with the [[Python Dictionary#^543235|string]] 'Unknown'?
titanic_data['Cabin'].fillna('Unknown')
```

### iloc

**Explanation:** The iloc() function is used for integer-location-based indexing and slicing.

```python
titanic_data.iloc[0]

# just on rows
titanic_data.iloc[:1]

# on rows and columns
titanic_data.iloc[0:3]

# on multiple rows and columns
titanic_data.iloc[0:3,0:2]

# How would you select the last row and the last column of the DataFrame using iloc()?

titanic_data.shape

titanic_data.tail()

titanic_data.iloc[:1,:4]

titanic_data.iloc[890,11]

titanic_data.columns
```

### Columns

**Explanation:** the columns function will extract all the columns from the relevant data

### Lowercase Column Names

**Explanation:** The str.lower() function is applied to the column names to convert them to lowercase.

```python
# lower columns
titanic_data.columns.str.lower()

# how to change the columns in the DataFrame
titanic_data.columns = titanic_data.columns.str.lower()

# add your answer here
# titanic_data.columns = titanic_data.columns.str.upper()

titanic_data.columns.str.upper()
```

```python

```