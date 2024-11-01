# This code is based on the work of Morgenroth et al. (2020) (DOI: 10.3390/f11020135)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew
import matplotlib.lines as mlines

## Set up the model
# Import data and merge
master_df = pd.read_csv(r'C:\Users\alexj\Documents\Research\Canadian Urban Forest Inventories - Structure and Diversity\Python Scripts and Datasets\(2) Filtered Master Dataset.csv', low_memory=False)
location_index_df = pd.read_csv(r'C:\Users\alexj\Documents\Research\Canadian Urban Forest Inventories - Structure and Diversity\Python Scripts and Datasets\Non-Inventory Datasets\Location Index.csv', low_memory=False)
df = master_df.merge(location_index_df, how='left', on='City')
excluded_cities = ['Maple Ridge', 'New Westminster', 'Peterborough', 'Halifax']
df = df[~df['City'].isin(excluded_cities)]

# Ensure DBH is numeric and drop NaN values
df['DBH'] = pd.to_numeric(df['DBH'], errors='coerce')
df = df.dropna(subset=['DBH'])

# Binning Information
bins = [0, 20, 40, 60, float('inf')]
bin_midpoints = [10, 30, 50, 70]
labels = ['0-20', '20-40', '40-60', '60+']
Type_3 = [0.25, 0.25, 0.25, 0.25]

## Find the median DBH for each city
median_dbh_per_city = df.groupby('City')['DBH'].median().reset_index()
print("Median DBH per City:")
print(median_dbh_per_city)
print("\nCity Skewness:")
city_skewness = df.groupby('City')['DBH'].apply(lambda x: skew(x, nan_policy='omit')).reset_index()
print(city_skewness)

## Structural Diversity Index
# Calculate Shannon-Wiener index for each city
def calculate_shannon_wiener_city(city_data, bins):
    # Classify trees into DBH classes using binning
    dbh_classes = pd.cut(city_data['DBH'], bins=bins, labels=range(1, len(bins)))

    # Calculate the proportion of trees in each class
    class_counts = dbh_classes.value_counts(normalize=True).sort_index()

    # Check for empty upper bins and adjust the number of classes
    last_non_empty_class = class_counts[class_counts > 0].index[-1]
    valid_bins = bins[:last_non_empty_class + 1]  # Include up to the last non-empty class

    # Reclassify using valid bins
    dbh_classes = pd.cut(city_data['DBH'], bins=valid_bins, labels=range(1, len(valid_bins)))
    class_counts = dbh_classes.value_counts(normalize=True).sort_index()

    # Filter out zero proportions to avoid log(0)
    class_proportions = class_counts[class_counts > 0]

    # Calculate the Shannon-Wiener index (H)
    H = -np.sum(class_proportions * np.log(class_proportions))

    # Calculate H_max (maximum possible diversity)
    H_max = np.log(len(valid_bins) - 1)  # Use the number of non-empty bins

    return H, H_max

city_results = []

cities = df['City'].unique()
for city in cities:
    city_data = df[df['City'] == city]

    if len(city_data) == 0:
        continue  # Skip if there are no valid DBH values for the city

    H, H_max = calculate_shannon_wiener_city(city_data, bins)

    # Store the result for the city
    city_results.append({
        'City': city,
        'Shannon-Wiener Index (H)': H,
        'Maximum Diversity (H_max)': H_max
    })

# Convert results to a DataFrame for easy viewing
Structural_Diversity_Index_city_df = pd.DataFrame(city_results)
print("\nStructural Diversity Index per City:")
print(Structural_Diversity_Index_city_df)

## Compare the Distributions to Type I, II, and III points for comparison
# Bin the real data
df['DBH_bin'] = pd.cut(df['DBH'], bins=bins, labels=labels,
                       right=False)  # Add a new column 'DBH_bin' that categorizes 'DBH' into the bins
grouped = df.groupby(['City', 'DBH_bin']).size().unstack(fill_value=0)  # Group by 'City' and 'DBH_bin' and count the occurrences
proportions = grouped.div(grouped.sum(axis=1), axis=0)  # Calculate the proportion of trees in each bin for every city
print("\nProportions of Trees in Each Bin per City:")
print(proportions)

# Define Jensen-Shannon Divergence Function
def calculate_js_divergence(P, Q):
    # Ensure the distributions are numpy arrays
    P = np.array(P)
    Q = np.array(Q)
    # Add a small epsilon to avoid log(0)
    epsilon = 1e-10
    P = P + epsilon
    Q = Q + epsilon
    P = P / np.sum(P)
    Q = Q / np.sum(Q)
    M = 0.5 * (P + Q)
    js_divergence = 0.5 * np.sum(P * np.log(P / M)) + 0.5 * np.sum(Q * np.log(Q / M))
    return js_divergence

# Define Type 1 (Exponential Decay) and Type 2 (Gaussian) functions
def exponential_decay(x, a, b):
    return a * np.exp(-b * x)

def gaussian(x, std_dev):
    return (1 / (std_dev * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - 30) / std_dev) ** 2)

# Adjusted optimize_type_1 function
def optimize_type_1(city_proportions):
    actual_proportions = city_proportions.values

        # Define ranges for a and b
    a_values = np.linspace(0.1, 5.0, 50)
    b_values = np.linspace(0.01, 1.0, 50)

    best_js_divergence = np.inf
    best_a = None
    best_b = None

    for a in a_values:
        for b in b_values:
            predicted_proportions = exponential_decay(np.array(bin_midpoints), a, b)
            predicted_proportions = predicted_proportions / np.sum(predicted_proportions)
            js_divergence = calculate_js_divergence(actual_proportions, predicted_proportions)
            if js_divergence < best_js_divergence:
                best_js_divergence = js_divergence
                best_a = a
                best_b = b

    return best_js_divergence, best_a, best_b

# Adjusted optimize_type_2 function
def optimize_type_2(city_proportions):
    actual_proportions = city_proportions.values

    # Define range for std_dev
    std_dev_values = np.linspace(5, 50, 50)

    best_js_divergence = np.inf
    best_std_dev = None

    for std_dev in std_dev_values:
        predicted_proportions = gaussian(np.array(bin_midpoints), std_dev)
        predicted_proportions = predicted_proportions / np.sum(predicted_proportions)
        js_divergence = calculate_js_divergence(actual_proportions, predicted_proportions)
        if js_divergence < best_js_divergence:
            best_js_divergence = js_divergence
            best_std_dev = std_dev

    return best_js_divergence, best_std_dev

# Dictionary to store the optimized JSD values and parameters
optimized_results = {
    'City': [],
    'JS Divergence Type 1': [],
    'Type 1 Best a': [],
    'Type 1 Best b': [],
    'JS Divergence Type 2': [],
    'Type 2 Best std_dev': [],
    'JS Divergence Type 3': [],
    'Best_Fit': []
}

# Iterate over cities and calculate optimized JSD for Type 1 and Type 2
for city in proportions.index:
    city_data = proportions.loc[city]

    # Optimize for Type 1 (Exponential Decay)
    type_1_js_divergence, best_a, best_b = optimize_type_1(city_data)

    # Optimize for Type 2 (Gaussian)
    type_2_js_divergence, best_std_dev = optimize_type_2(city_data)

    # Calculate JS Divergence for Type 3 (Equal Distribution)
    type_3_js_divergence = calculate_js_divergence(city_data.values, Type_3)

    # Determine the best fit type
    js_divergences = {'Type 1': type_1_js_divergence, 'Type 2': type_2_js_divergence, 'Type 3': type_3_js_divergence}
    best_fit = min(js_divergences, key=js_divergences.get)

    # Append the results for the city
    optimized_results['City'].append(city)
    optimized_results['JS Divergence Type 1'].append(type_1_js_divergence)
    optimized_results['Type 1 Best a'].append(best_a)
    optimized_results['Type 1 Best b'].append(best_b)
    optimized_results['JS Divergence Type 2'].append(type_2_js_divergence)
    optimized_results['Type 2 Best std_dev'].append(best_std_dev)
    optimized_results['JS Divergence Type 3'].append(type_3_js_divergence)
    optimized_results['Best_Fit'].append(best_fit)

# Convert to DataFrame for better display
optimized_js_divergence_df = pd.DataFrame(optimized_results)
optimized_js_divergence_df.to_csv(r'optimized_js_divergence_df.csv', index=False)

# Display the DataFrame
pd.set_option('display.max_columns', None)
print("\nOptimized JS Divergence and Best Fit Parameters per City:")
print(optimized_js_divergence_df)

# Iterate over each city and plot the actual proportions against the best-fit distribution
for city in proportions.index:
    city_data = proportions.loc[city].values
    best_fit_type = optimized_js_divergence_df[optimized_js_divergence_df['City'] == city]['Best_Fit'].values[0]

    # Prepare the best fit line based on the optimization results
    if best_fit_type == 'Type 1':
        best_a = optimized_js_divergence_df[optimized_js_divergence_df['City'] == city]['Type 1 Best a'].values[0]
        best_b = optimized_js_divergence_df[optimized_js_divergence_df['City'] == city]['Type 1 Best b'].values[0]
        best_fit_proportions = exponential_decay(np.array(bin_midpoints), best_a, best_b)
        best_fit_proportions = best_fit_proportions / np.sum(best_fit_proportions)
    elif best_fit_type == 'Type 2':
        best_std_dev = optimized_js_divergence_df[optimized_js_divergence_df['City'] == city]['Type 2 Best std_dev'].values[0]
        best_fit_proportions = gaussian(np.array(bin_midpoints), best_std_dev)
        best_fit_proportions = best_fit_proportions / np.sum(best_fit_proportions)  # Normalize Gaussian
    else:
        best_fit_proportions = Type_3

    # Normalize best-fit proportions
    best_fit_proportions = best_fit_proportions / np.sum(best_fit_proportions)

    # Plot the actual proportions and the best-fit line
    plt.figure(figsize=(8, 6))
    plt.plot(bin_midpoints, city_data, label=f"{city} Actual", marker='o', color='blue')
    plt.plot(bin_midpoints, best_fit_proportions, label=f"Best Fit ({best_fit_type})", marker='o', linestyle='--', color='red')
    plt.title(f"{city}: Actual vs Best-Fit Distribution")
    plt.xlabel("DBH Bin Midpoints")
    plt.ylabel("Proportion")
    plt.legend()
    plt.grid(True)

    # Show the plot for each city
    plt.show()

# Plotting the distributions
# Type 1 - average of best_a and best_b from all cities
average_a = optimized_js_divergence_df['Type 1 Best a'].mean()
average_b = optimized_js_divergence_df['Type 1 Best b'].mean()
exp_decay_line = exponential_decay(np.array(bin_midpoints), average_a, average_b)
exp_decay_line = exp_decay_line / np.sum(exp_decay_line)

# Type 2 - use average std_dev
average_std_dev = optimized_js_divergence_df['Type 2 Best std_dev'].mean()
Type_2_gaussian_line = gaussian(np.array(bin_midpoints), average_std_dev)
Type_2_gaussian_line = Type_2_gaussian_line / np.sum(Type_2_gaussian_line)

plt.figure(figsize=(8, 6))
plt.plot(bin_midpoints, exp_decay_line, label="Type 1", marker='o')
plt.plot(bin_midpoints, Type_2_gaussian_line, label="Type 2", marker='o')
plt.plot(bin_midpoints, Type_3, label="Type 3", marker='o')
plt.title("Comparison of Distributions (Type 1, Type 2, Type 3)")
plt.xlabel("DBH Bin Midpoints")
plt.ylabel("Proportion")
plt.legend()
plt.grid(True)
plt.show()