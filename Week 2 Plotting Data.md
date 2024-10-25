# Tags
#python #VisualisingData

## Pivot tables/ pd.pivot_table

Default function
>[!info] pandas.pivot_table
>(**_data_**, **_values**=None_, **_index**=None_, **_columns**=None_, **_aggfunc**='mean'_, 
>**_fill_value**=None_, **_margins**=False_, **_dropna**=True_, **_margins_name**='All'_, _
>**observed**=_NoDefault.no_default_, **_sort**=True_\

Creates a spreadsheet-style pivot table as a DataFrame.

The levels in the pivot table will be stored in MultiIndex objects (hierarchical indexes) on the [[Python Dictionary#^2fb0a5|Index]] and columns of the result DataFrame.

[[Python Dictionary#^2fb0a5|Index]]

**Pivot for the average fare of survival rate by gender**

```python
pivot_avg_fare_by_sex = data.pivot_table(values='Fare', index='Sex', columns='Survived', aggfunc='mean')

pivot_avg_fare_by_sex

#values = values that you will be looking at
#aggfunc="") takes the value can be sum,avg,etc...
#index = rows 
#columns = columns (optional function)

```
![[Pasted image 20240906151439.png]]
## Pandas Plot and Matplotlib
```python
# import libraries
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('train.csv')

# Plot a bar chart of the number of survivors
data['Survived'].value_counts().plot(kind='bar')
```

![[Pasted image 20240905133527.png]]

This is the fastest way to create a chart using pandas.

But not all data is this simple to visualise nor is this chart the best way to visualise the chart.

**What changes would you make to this chart to make it clearer to understand?**

```python
# Plot a bar chart of the number of survivors
data['Survived'].value_counts().plot(kind='bar', 
                    color=['blue', 'orange'], 
                    # adding in colour
                    figsize=(20,20) 
                    # changing the size
                    )
```

![[Pasted image 20240905133644.png]]
**What else is missing from this chart?**

To add finer details like chart titles, axis labels etc we need to use the matplotlib library to make these changes.

```python
# Plot a bar chart of the number of survivors
plt.figure(figsize=(8, 10))
survival_counts = data['Survived'].value_counts()
survival_counts.plot(kind='bar', color=['darkblue', 'green'])

# adding a title to the chart
plt.title('Number of Survivors', fontsize=30)

# adding an x axis and a label
plt.xlabel('Survived', fontsize=25)

# adding a y axis and a label
plt.ylabel('Count', fontsize=25)

# adding font sizes to the axis counts
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
```

![[Pasted image 20240905133803.png]]

##### Pandas and Matplot Examples

```python
# import relevant libraries
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('train.csv')

# Create a pivot table of survival rate by sex
survival_by_sex = data.pivot_table(index='Sex', columns='Survived', values='PassengerId', aggfunc='count')

# Plot the pivot table
plt.figure(figsize=(20, 20))
survival_by_sex.plot(kind='bar', stacked=True, color=['purple', 'green'])

plt.title('Survival Count by Sex', fontsize=20)
plt.xlabel('Sex', fontsize=15)
plt.ylabel('Count', fontsize=15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()
```
![[Pasted image 20240905135228.png]]

```python
# Pivot table for the average age of survival rate by gender
pivot_avg_age_by_gender = data.pivot_table(values='Age', index='Sex', columns='Survived', aggfunc='mean')

# Plotting the pivot table
pivot_avg_age_by_gender.plot(kind='bar', figsize=(20, 20), color=['lightcoral', 'lightseagreen'])
plt.title('Average Age of Survival Rate by Gender', fontsize=20)
plt.xlabel('Gender', fontsize=15)
plt.ylabel('Average Age', fontsize=15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(['Not Survived', 'Survived'], fontsize=15)
plt.show()

```
![[Pasted image 20240905184223.png]]

```python
# Pivot table for the average fare of survival rate by gender
pivot_avg_fare_by_gender = data.pivot_table(values='Fare', index='Sex', columns='Survived', aggfunc='mean')

# Plotting the pivot table
pivot_avg_fare_by_gender.plot(kind='bar', figsize=(20, 20), color=['mediumpurple', 'gold'])
plt.title('Average Fare of Survival Rate by Gender', fontsize=20)
plt.xlabel('Gender', fontsize=15)
plt.ylabel('Average Fare', fontsize=15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(['Not Survived', 'Survived'], fontsize=15)
plt.show()
```
![[Pasted image 20240905184354.png]]

```python
# Pivot table for the average fare of survival rate by class
pivot_avg_fare_by_class = data.pivot_table(values='Fare', index='Pclass', columns='Survived', aggfunc='mean')

# Plotting the pivot table
pivot_avg_fare_by_class.plot(kind='bar', figsize=(20, 20), color=['steelblue', 'darkorange'])
plt.title('Average Fare of Survival Rate by Class', fontsize=20)
plt.xlabel('Pclass', fontsize=15)
plt.ylabel('Average Fare', fontsize=15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(['Not Survived', 'Survived'], fontsize=15)
plt.show()
```
![[Pasted image 20240905184616.png]]

```python
# Creating a pivot table for the average fare of survival rate by class and gender
pivot_avg_fare_by_class_gender = data.pivot_table(values='Fare', index=['Pclass', 'Sex'], columns='Survived', aggfunc='mean')

# Plotting the pivot table
pivot_avg_fare_by_class_gender.plot(kind='bar', figsize=(20, 20), color=['cornflowerblue', 'lightcoral'])
plt.title('Average Fare of Survival Rate by Class and Gender', fontsize=20)
plt.xlabel('Class and Gender', fontsize=15)
plt.ylabel('Average Fare', fontsize=15)
plt.xticks(fontsize=12, rotation=45)
plt.yticks(fontsize=12)
plt.legend(['Not Survived', 'Survived'], fontsize=15)
plt.show()
```

![[Pasted image 20240905184759.png]]

```python
# import library
import pandas as pd
import matplotlib.pyplot as plt

# import data
data = pd.read_csv('train.csv')

plt.figure(figsize=(6, 6))
plt.scatter(data['Age'], data['Fare'], c='#5d0964',s=100,alpha=0.6) 

# Add the relevant titles and font size updates
plt.title('Fare over age', fontsize=20)
plt.xlabel('Age', fontsize=15)
plt.ylabel('Fare', fontsize=15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()
```
![[Pasted image 20240905191600.png]]


## Matplotlib overview
**Overview:**

- Matplotlib is a comprehensive library for creating static, animated, and interactive visualizations in Python.
    

**Key Features:**

- **Extensive Customization:** Offers detailed control over every aspect of the plot, from colors and labels to ticks and grid lines.
    
- **Wide Range of Plot Types:** Supports a vast [[Python Dictionary#^d8476a|array]] of plot types, including line, bar, scatter, histogram, and more complex plots like 3D and polar charts.
    
- **Publication-Quality Figures:** Capable of producing high-quality plots suitable for publication.
    

**Advantages:**

- **High Customizability:** Allows fine-grained control over the appearance of plots.
    
- **Versatility:** Can create a wide range of plot types, from simple to highly complex.
    
- **Large Community:** Well-documented with a large community, offering extensive resources and examples.
    

**Disadvantages:**

- **Complexity:** Can be more complex and verbose compared to Pandas.plot() and Seaborn.
    
- **Steep Learning Curve:** Requires more time to learn and master, especially for complex customizations.

## Seaborn
**Overview:**

- Seaborn is a statistical data visualization library based on Matplotlib, designed to create informative and attractive visualizations with less effort.
    

**Key Features:**

- **Statistical Plots:** Includes built-in themes and color palettes for statistical plots, such as distribution plots, box plots, and violin plots.
    
- **Easy Aesthetics:** Automatically styles the plots with aesthetically pleasing defaults.
    
- **Integration with Pandas:** Works well with Pandas dataframes, allowing for easy plotting directly from dataframes.
    

**Advantages:**

- **Beautiful Plots:** Generates visually appealing and informative plots with minimal code.
    
- **Simplified Syntax:** Provides higher-level interfaces for drawing attractive and informative statistical graphics.
    
- **Built-in Themes:** Comes with built-in themes and color palettes, enhancing the aesthetics of the plots.
    

**Disadvantages:**

- **Less Control:** Less flexible than Matplotlib for detailed customizations.
    
- **Dependency on Matplotlib:** Relies on Matplotlib, so some advanced features may still require Matplotlib adjustments.

##### Seaborn Examples

```python
# import libraries
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# import data
data = pd.read_csv('train.csv')


# Plot a histogram of passenger ages
plt.figure(figsize=(20, 20))
sns.histplot(data['Age'].dropna(), # what is the dropna doing?
             kde=True, # adds a kernel density estimate to the plot
             color='skyblue')

# Update titles and axis
plt.title('Distribution of Passenger Ages', fontsize=20)
plt.xlabel('Age', fontsize=15)
plt.ylabel('Frequency', fontsize=15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()
```
![[Pasted image 20240905191903.png]]

```python
# Plot a bar plot showing the number of survivors by class and sex
plt.figure(figsize=(20, 20))
sns.barplot(x='Pclass',
            y='Survived',
            hue='Sex',
            data=data,
            palette='Set1')

plt.title('Survival Rate by Class and Sex', fontsize=20)
plt.xlabel('Class', fontsize=15)
plt.ylabel('Survival Rate', fontsize=15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(title='Sex', fontsize=12, title_fontsize=15)
plt.show()
```
![[Pasted image 20240905191940.png]]
###### Pair Plot
```python
# use seaborn's pairplot with a hue for gender
sns.pairplot(df, hue="Gender")
```
![[Pasted image 20240905192809.png]]