# Insights and potential actions that can be taken from them

## Importing and EDA


```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
```

```python
# Load the airbnb data airbnb_df = 

pd.read_csv('airbnb_data.csv')
```

``` python
# Load the location data
location_df = pd.read_csv('location.csv')
```

```python
# merge the datasets 
df = pd.merge(airbnb_df,location_df,
                 left_on='id',right_on='id',how='inner')

```

```python
# EDA 
df.info()

df.describe()
```

## Room type Analysis

**Room Type Distribution:** Entire home/apartments dominate the listings, which could indicate a shift towards more professional hosting and potentially impact long-term housing availability.

- Actionable Insight: Local authorities might want to investigate if this high proportion of entire home listings is affecting the local housing market and consider implementing or adjusting regulations.

```python
# Room type distribution
room_type_counts = df['room_type'].value_counts()
plt.figure(figsize=(10, 6))
room_type_counts.plot(kind='bar')
plt.title('Distribution of Room Types')
plt.xlabel('Room Type')
plt.ylabel('Count')
plt.show()
```
![[Pasted image 20241011115559.png]]

## Price Analysis 
**Price Analysis: 

There's likely a significant price difference between room types, with entire homes/apartments being the most expensive.

- Actionable Insight: Hosts could optimize their pricing strategy based on room type and local averages. For budget-conscious travelers, private rooms might offer the best value.
```python
# Price analysis
plt.figure(figsize=(10, 6))
sns.boxplot(x='room_type', y='price', data=df)
plt.title('Price Distribution by Room Type')
plt.xlabel('Room Type')
plt.ylabel('Price')
plt.show()
```
![[Pasted image 20241011115622.png]]

### How do we make the boxplot easier to understand?


``` python
# Price analysis with outlier removal
plt.figure(figsize=(12, 6))

# Calculate Q1, Q3, and IQR for each room type
Q1 = df.groupby('room_type')['price'].transform('quantile', 0.25)
Q3 = df.groupby('room_type')['price'].transform('quantile', 0.75)
IQR = Q3 - Q1
```

- Q1 (First Quartile): This is the 25th percentile of the data. It's calculated for the 'price' column, separately for each 'room_type'.
    
- Q3 (Third Quartile): This is the 75th percentile, also calculated for each room type.
    
- IQR (Interquartile Range): This is the range between Q1 and Q3, representing the middle 50% of the data.
    

The transform function is doing something special here.

- It's like a smart copy-paste. It calculates a value for each group (room type in our case) and then pastes that value next to every row in that group. This makes our next steps much easier because everything lines up perfectly.

Example - let's say we have this tiny dataset

```python
# Define bounds for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
```

These lines define the lower and upper bounds for what we consider "normal" data.

Any data point below Q1 - 1.5 * IQR or above Q3 + 1.5 * IQR is considered an outlier.

The factor 1.5 is a common choice in statistics for identifying outliers, but it can be adjusted based on how strict you want to be.

- Any data point that is more than 1.5 times the IQR below Q1 or above Q3 is considered an outlier.
- This creates a range of "acceptable" values, beyond which data points are flagged as unusual.

Why 1.5?

- This value was proposed by John Tukey, a famous statistician, as part of his method for identifying outliers.
- It's based on statistical properties of the normal distribution.
- For normally distributed data, this rule identifies about 1% of the data as outliers.

``` python
# Create a mask for non-outlier data
mask = (df['price'] >= lower_bound) & (df['price'] <= upper_bound)
```

**Creating a boolean mask:**

Think of a mask like a list of "yes" or "no" answers for each price in our data.

- "Yes" (or True) means the price is normal (not an outlier).
- "No" (or False) means the price is unusual (an outlier).

**How it works:**

For each price, we ask two questions:

- Is it higher than or equal to the lower bound?
- Is it lower than or equal to the upper bound?
- If the answer to both questions is "yes", we keep that price (it's not an outlier).
- If the answer to either question is "no", we don't keep that price (it's an outlier).

**The '&' symbol:**

- This is just a way to combine our two questions.
- It's like saying "AND" between the questions.
- We only keep prices that pass both checks.

**Using the mask:**

- When we make our boxplot, we use this list of "yes" and "no" answers.
- We only show the prices that got a "yes" (True) in our mask.
- This way, we leave out the unusual prices when drawing our chart.

```python
# Create the boxplot with outliers removed
sns.boxplot(x='room_type', y='price', data=df[mask])
plt.title('Price Distribution by Room Type (Outliers Removed)')
plt.xlabel('Room Type')
plt.ylabel('Price')
# plt.yscale('log')  # Use log scale for better visualization
plt.show()
```
![[Pasted image 20241011120011.png]]

#  Log Scale 

**What's a log scale do?**

plt.yscale('log') changes how we show numbers on the y-axis (vertical axis) of our graph.

**Normal scale vs. Log scale:**

- Normal scale: Numbers are spaced evenly (like 0, 10, 20, 30, 40, 50...)
- Log scale: Numbers increase by multiplication (like 1, 10, 100, 1000...)

**Why use it:**

- It's helpful when you have a wide range of numbers, from very small to very large.
- It makes it easier to see patterns across different price ranges.

**How it helps:**

- Small differences in lower prices become more visible.
- Large price differences at the high end don't overwhelm the chart.

+ Code+ Text

### Try the first boxplot again with a log scale

```python
# Price analysis
plt.figure(figsize=(10, 6))
sns.boxplot(x='room_type', y='price', data=df)
plt.title('Price Distribution by Room Type')
plt.xlabel('Room Type')
plt.yscale('log')  # Use log scale for better visualization
plt.ylabel('Price')
plt.show()
```
![[Pasted image 20241011120114.png]]

## Availability and Price Analysis
**Availability vs. Price**: This analysis can reveal if there's a correlation between price and availability. Higher-priced listings might have lower availability, indicating they're more in demand or intentionally listed for shorter periods.

- Actionable Insight: Hosts could adjust their prices or minimum stay requirements based on their desired occupancy rate.
```python
# Minimum Nights analysis
plt.figure(figsize=(10, 6))
sns.scatterplot(x='minimum_nights', y='price', hue='room_type', data=df)
plt.title('Price vs Minimum Nights')
plt.xlabel('Price')
plt.ylabel('Minimum Nights')
plt.show()
```
![[Pasted image 20241011120156.png]]

### How can we make the scatter plot easier to read?

```python
# Minimum Nights analysis
plt.figure(figsize=(10, 6))
sns.scatterplot(x='minimum_nights', y='price', hue='room_type', data=df)
plt.title('Price vs Minimum Nights')
# plt.xscale('log')  # Use log scale for minimum_nights
plt.yscale('log')  # Use log scale for price
plt.xlabel('Price')
plt.ylabel('Minimum Nights')
plt.show()

```

![[Pasted image 20241011120251.png]]

```python
# Minimum Nights analysis
plt.figure(figsize=(10, 6))
sns.scatterplot(x='minimum_nights', y='price', hue='neighbourhood', data=df)
plt.title('Price vs Minimum Nights')
# plt.xscale('log')  # Use log scale for minimum_nights
plt.yscale('log')  # Use log scale for price
plt.xlabel('Price')
plt.ylabel('Minimum Nights')
plt.show()
```
![[Pasted image 20241011120351.png]]

## Muli-listing Host Analysis

**Multi-listing Hosts**: The number of hosts with multiple listings indicates the level of professionalization in the local Airbnb market.

- Actionable Insight: A high number of multi-listing hosts might prompt authorities to investigate if these are operating as de facto hotels without proper licenses or tax compliance.

```python
# Host analysis (Who has multiple listing)
host_listing_counts = df['host_id'].value_counts()
multi_listing_hosts = host_listing_counts[host_listing_counts > 1].count()
print(f"Number of hosts with multiple listings: {multi_listing_hosts}")
```

## Neighborhood Analysis
**Neighborhood Analysis**: Identifies the most popular neighborhoods for Airbnb listings.

- Actionable Insight: City planners and local businesses can use this information to anticipate tourist flows and adjust services accordingly. It may also highlight areas where housing pressure from short-term rentals is most acute.

```python
# Neighborhood analysis
top_neighborhoods = df['neighbourhood'].value_counts().head(10)
plt.figure(figsize=(12, 6))
top_neighborhoods.plot(kind='bar')
plt.title('Top 10 Neighborhoods by Listing Count')
plt.xlabel('Neighborhood')
plt.ylabel('Number of Listings')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
```
![[Pasted image 20241011120517.png]]

```python
# Create the pivot table
pivot_data = df.pivot_table(index='neighbourhood', columns='room_type',
                            values='id', aggfunc='count')

# Sort the data by total listings in each neighborhood
pivot_data_sorted = pivot_data.sort_values(by=pivot_data.columns.tolist(), ascending=False)

# Set up the plot style
try:
    plt.style.use('ggplot')
except:
    print("ggplot style not available, using default style.")

# Create the figure and axes
fig, ax = plt.subplots(figsize=(14, 8))  # Increased width to accommodate legend

# Create the stacked bar plot
pivot_data_sorted.plot(kind='bar', stacked=True, ax=ax, 
                       colormap='Set2')  # Using a colormap that's color-blind friendly

# Customize the plot
ax.set_title('Distribution of Airbnb Listings by Neighborhood and Room Type', fontsize=16)
ax.set_xlabel('Neighborhood', fontsize=12)

ax.set_ylabel('Number of Listings', fontsize=12)

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha='right')

# Add and customize the legend
legend = ax.legend(title='Room Type', loc='center left', bbox_to_anchor=(1, 0.5))
plt.setp(legend.get_title(), fontsize='12')  # Adjust legend title font size
plt.setp(legend.get_texts(), fontsize='10')  # Adjust legend text font size

# Adjust layout to make room for the legend
fig.subplots_adjust(right=0.85)

# Ensure x-axis labels are not cut off
fig.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust the right boundary to prevent legend cutoff

# Show the plot
plt.show()
```

![[Pasted image 20241011120623.png]]

## License Analysis
**License Analysis**: Shows the compliance rate with local regulations.

- Actionable Insight: If there's a significant number of unlicensed or exempt listings, authorities may need to increase enforcement efforts or clarify regulations.
```python
# License analysis
license_counts = df['license'].value_counts()
print("License distribution:")
print(license_counts)
```

## Average Metrics by Room Type

**Average Metrics by Room Type**: Provides a clear comparison of price, availability, and popularity (number of reviews) across different room types.

- Actionable Insight: Hosts can use this to benchmark their listings and adjust their offerings to match or beat the average for their room type.
### Group by to calculate averages

```python
# Calculate average price and availability for different room types
avg_metrics = df.groupby('room_type').agg({
    'price': 'mean',
    'availability_365': 'mean',
    'number_of_reviews': 'mean'
}).round(2)

print("\nAverage metrics by room type:")
print(avg_metrics) 
```

![[Pasted image 20241011120923.png]]

## Commercial Operators Analysis  

**Potential Commercial Operators**: Identifies hosts with the most listings, which could indicate commercial operations.

- Actionable Insight: Regulators might want to scrutinize these hosts to ensure they're complying with all relevant hospitality and housing laws.

```python
# Identify potential commercial operators (hosts with many listings)
potential_commercial = df['host_name'].value_counts().head(10)
print("\nTop 10 hosts by number of listings:")
print(potential_commercial)

```

![[Pasted image 20241011120837.png]]

# Running A Logistical Regression/Random Forest to predict Price

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt

```


```python
# Load the airbnb data
airbnb_df = pd.read_csv('airbnb_data.csv')

# Load the location data
location_df = pd.read_csv('location.csv')

# merge the datasets 
df = pd.merge(airbnb_df,location_df,
                 left_on='id',right_on='id',how='inner')
```

```python 
#EDA 
df.head()
```

```python
# Select features for prediction, now including neighbourhood
features = ['room_type', 'neighbourhood', 'minimum_nights', 'number_of_reviews', 
            'reviews_per_month', 'calculated_host_listings_count', 'availability_365']

# Create a new dataframe with only the selected features and price
df_selected = df[features + ['price']]

# Drop rows with NaN values
df_clean = df_selected.dropna()

# Convert categorical variables to dummy variables
df_encoded = pd.get_dummies(df_clean, columns=['room_type', 'neighbourhood'])

# Separate features (X) and target variable (y)
X = df_encoded.drop('price', axis=1)
y = df_encoded['price']
```

```python
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

```python
# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```


### Why are we scaling the data?

This code is performing a process called feature scaling, specifically using a method called standardization.

**Why do we do this?**

- Imagine you have data about houses: number of rooms (1-10) and price (100,000−100,000−1,000,000).
- These numbers are on very different scales.
- Scaling makes all features have similar importance in the model's "eyes".

**What does it actually do?**

- For each feature, it subtracts the average and divides by the standard deviation.
- This makes each feature have an average of 0 and a standard deviation of 1.

**Benefits:**

- It helps many machine learning algorithms perform better.
- It makes features comparable to each other.
- It can speed up the learning process for some algorithms.

In simple terms, it's like converting all our measurements to a standard unit, so our model can easily compare and use them, regardless of their original scale.

```python
# Linear Regression Model
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
lr_predictions = lr_model.predict(X_test_scaled)
```

```python
# Random Forest Model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)
```

```python
# Evaluation function using Mean Absolute Percentage Error (MAPE)
def evaluate_model(y_true, y_pred, model_name):
    mape = mean_absolute_percentage_error(y_true, y_pred)
    print(f"{model_name} - Mean Absolute Percentage Error: {mape:.2%}")
```

```python
evaluate_model(y_test, lr_predictions, "Linear Regression")
evaluate_model(y_test, rf_predictions, "Random Forest")
```

```python
# Visualize predictions vs actual
plt.figure(figsize=(8, 5))
plt.scatter(y_test, lr_predictions, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Linear Regression: Actual vs Predicted")

plt.tight_layout()
plt.show()
```


![[Pasted image 20241011123221.png]]

```python
# Visualize predictions vs actual
plt.figure(figsize=(8, 5))
plt.scatter(y_test, rf_predictions, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Random Forest: Actual vs Predicted")

plt.tight_layout()
plt.show()
```
![[Pasted image 20241011123315.png]]



# Using Mean Absolute Percentage Error
We've chosen to use the Mean Absolute Percentage Error (MAPE) as our primary evaluation metric. Let's go into why MAPE is the most appropriate metric for this Airbnb price prediction task:

- **Interpretability**: MAPE expresses the error as a percentage of the actual value, making it highly interpretable. For instance, a MAPE of 15% means that, on average, our predictions are off by 15% of the actual price. This is easily understood by both technical and non-technical stakeholders in the Airbnb ecosystem.
    
- **Scale-Independence**: Airbnb prices can vary widely, from budget accommodations to luxury properties. MAPE is scale-independent, meaning it works well across this wide range of prices. It allows for fair comparison of prediction accuracy whether we're dealing with a $50 per night room or a $500 per night apartment.
    
- **Business Relevance:** In the context of Airbnb pricing, percentage errors are more relevant than absolute errors. A $10 error on a $50 listing is more significant than a $10 error on a $500 listing. MAPE captures this relative importance, aligning well with business perspectives on pricing accuracy.
    
- **Symmetry in Over-** and Under-predictions: MAPE treats over-predictions and under-predictions symmetrically. This is crucial for Airbnb pricing, where both overpricing (potentially losing bookings) and underpricing (potentially losing revenue) are important concerns.
    
- **Actionable Insights:** MAPE provides clear, actionable insights. If a model has a MAPE of 20%, Airbnb hosts and the platform can understand that they should expect price recommendations to be within approximately 20% of the optimal price, guiding their pricing strategies and expectations.
    
- **Comparability Across Models**: MAPE allows for easy comparison between different models (like our Linear Regression and Random Forest models) and even across different datasets or markets, as it's a relative measure.
    
- **Robustness to Outliers**: While not completely immune to outliers, MAPE is less sensitive to extreme values compared to metrics like Mean Squared Error. This is beneficial in the Airbnb context where there might be some extremely high-priced luxury listings.
    

### Limitations to Consider:

- MAPE can be biased towards under-predictions, as it has a lower bound of 0% but no upper bound.
    
- It can't be used if there are actual zero values in the dataset, which is unlikely for Airbnb prices but worth noting.


- Talk about data (no outliers/low correlation between income and other variables)
- Introduce and talk about key insights.
- Talk about modeling (Data imputation, What the models do? , Why Am i using MAPE)
- Conclude. 