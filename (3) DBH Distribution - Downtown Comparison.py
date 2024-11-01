# This code is based on the work of Morgenroth et al. (2020) (DOI: 10.3390/f11020135)

import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu

## Import data and merge
master_df = pd.read_csv(r'C:\Users\alexj\Documents\Research\Canadian Urban Forest Inventories - Structure and Diversity\Python Scripts and Datasets\(2) Filtered Master Dataset.csv', low_memory=False)
downtown_df = pd.read_csv(r'C:\Users\alexj\Documents\Research\Canadian Urban Forest Inventories - Structure and Diversity\Python Scripts and Datasets\Non-Inventory Datasets\Downtown Areas.csv', low_memory=False)
df = master_df.merge(downtown_df, how='left', on='DAUID')
included_cities = ['Moncton', 'Fredericton', 'Quebec City', 'Longueuil', 'Montreal', 'Ottawa', 'Kingston',
 'Toronto', 'St. Catharines', 'Kitchener', 'Guelph', 'Windsor', 'Winnipeg', 'Regina', 'Lethbridge', 'Calgary',
 'Edmonton', 'Kelowna', 'Vancouver', 'Victoria', 'Mississauga', 'Burlington', 'Waterloo']
df = df[df['City'].isin(included_cities)]

# Define downtown and cities
df['DOWNTOWN'] = np.where(df['DOWNTOWN'] == 'Downtown', 1, 0)
cities = df['City'].unique()

# Ensure DBH is numeric and drop NaN values
df['DBH'] = pd.to_numeric(df['DBH'], errors='coerce')
df = df.dropna(subset=['DBH'])

# Group by City and Downtown and calculate median and standard deviation
grouped_stats = df.groupby(['City', 'DOWNTOWN'])['DBH'].agg(['median', 'std']).reset_index()
print(grouped_stats)

# Loop over each city and perform Kruskal-Wallis test for DBH grouped by Downtown
results = []
for city in cities:
 # Subset data for the current city
 city_data = df[df['City'] == city]

 # Group by Downtown (0 or 1)
 group_0 = city_data[city_data['DOWNTOWN'] == 0]['DBH']
 group_1 = city_data[city_data['DOWNTOWN'] == 1]['DBH']

 # Check for non-empty groups before performing the test
 if len(group_0) > 0 and len(group_1) > 0:
  # Perform Mann-Whitney U Test
  stat, p_value = mannwhitneyu(group_0, group_1, alternative='two-sided')

  # Append results
  results.append({'City': city, 'U-statistic': stat, 'p-value': p_value})
 else:
  # Handle cases where a group is empty
  results.append({'City': city, 'U-statistic': None, 'p-value': None})

# Convert results to DataFrame for easy viewing
results_df = pd.DataFrame(results)

# Export the DataFrame to a CSV file
grouped_stats.to_csv(r'(3) Downtown Comparison - Grouped Statistics.csv', index=False)
results_df.to_csv(r'(3) Downtown Comparison - Mann-Whitney U Test Results.csv', index=False)

## Best Fitting Distribution
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

# Define Type 1 (Exponential Decay), Type 2 (Gaussian), and Type 3 (Equal Distribution)
def exponential_decay(x, a, b):
    return a * np.exp(-b * x)

def gaussian(x, std_dev):
    return (1 / (std_dev * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - 30) / std_dev) ** 2)

Type_3 = [0.25, 0.25, 0.25, 0.25]  # Equal Distribution

# Define bins and bin midpoints for DBH
bins = [0, 20, 40, 60, float('inf')]
bin_midpoints = [10, 30, 50, 70]
labels = ['0-20', '20-40', '40-60', '60+']

# Function to optimize Type 1 (Exponential Decay)
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

# Function to optimize Type 2 (Gaussian)
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
    'Area': [],
    'JS Divergence Type 1': [],
    'Type 1 Best a': [],
    'Type 1 Best b': [],
    'JS Divergence Type 2': [],
    'Type 2 Best std_dev': [],
    'JS Divergence Type 3': [],
    'Best_Fit': []
}

# Compare downtown and non-downtown DBH distributions for each city
for city in cities:
    for area in [0, 1]:  # 0 for non-downtown, 1 for downtown
        city_data = df[(df['City'] == city) & (df['DOWNTOWN'] == area)]

        if len(city_data) == 0:
            continue  # Skip if there are no valid DBH values for the city

        # Bin the DBH data
        city_data['DBH_bin'] = pd.cut(city_data['DBH'], bins=bins, labels=labels, right=False)
        binned_counts = city_data['DBH_bin'].value_counts(normalize=True).sort_index()

        # Ensure all bins are represented
        binned_counts = binned_counts.reindex(labels, fill_value=0)

        # Optimize for Type 1 (Exponential Decay)
        type_1_js_divergence, best_a, best_b = optimize_type_1(binned_counts)

        # Optimize for Type 2 (Gaussian)
        type_2_js_divergence, best_std_dev = optimize_type_2(binned_counts)

        # Calculate JS Divergence for Type 3 (Equal Distribution)
        type_3_js_divergence = calculate_js_divergence(binned_counts.values, Type_3)

        # Determine the best fit type
        js_divergences = {'Type 1': type_1_js_divergence, 'Type 2': type_2_js_divergence, 'Type 3': type_3_js_divergence}
        best_fit = min(js_divergences, key=js_divergences.get)

        # Append the results for the city and area
        optimized_results['City'].append(city)
        optimized_results['Area'].append('Downtown' if area == 1 else 'Non-Downtown')
        optimized_results['JS Divergence Type 1'].append(type_1_js_divergence)
        optimized_results['Type 1 Best a'].append(best_a)
        optimized_results['Type 1 Best b'].append(best_b)
        optimized_results['JS Divergence Type 2'].append(type_2_js_divergence)
        optimized_results['Type 2 Best std_dev'].append(best_std_dev)
        optimized_results['JS Divergence Type 3'].append(type_3_js_divergence)
        optimized_results['Best_Fit'].append(best_fit)

# Convert to DataFrame for better display
optimized_js_divergence_df = pd.DataFrame(optimized_results)
optimized_js_divergence_df.to_csv(r'(3) Downtown Comparison - Optimized JSD.csv', index=False)

# Display the DataFrame
pd.set_option('display.max_columns', None)
print("\nOptimized JS Divergence and Best Fit Parameters per City and Area:")
print(optimized_js_divergence_df)