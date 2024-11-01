# This code is based on the work of Morgenroth et al. (2020) (DOI: 10.3390/f11020135)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

## Import data and merge
master_df = pd.read_csv(r'C:\Users\alexj\Documents\Research\Canadian Urban Forest Inventories - Structure and Diversity\Python Scripts and Datasets\(2) Filtered Master Dataset.csv', low_memory=False)
location_index_df = pd.read_csv(r'C:\Users\alexj\Documents\Research\Canadian Urban Forest Inventories - Structure and Diversity\Python Scripts and Datasets\Non-Inventory Datasets\Location Index.csv', low_memory=False)
df = master_df.merge(location_index_df, how='left', on='City')
excluded_cities = ['Maple Ridge', 'New Westminster', 'Peterborough', 'Halifax']
df = df[~df['City'].isin(excluded_cities)]

# Define cities and prepare lists for all midpoints and proportions
cities = df['City'].unique()
ecozone = df['Ecozone'].unique()
city_size = df['City Size'].unique()

## Clean and sort the DBH
# Ensure DBH is numeric and drop NaN values
df['DBH'] = pd.to_numeric(df['DBH'], errors='coerce')
df = df.dropna(subset=['DBH'])

## Comparison of Diameter Class Distributions Against Richards Distribution
plt.figure(figsize=(12, 6))

# Set font sizes explicitly by doubling the default values
plt.rcParams.update({
    'font.size': 14,  # General text size
    'axes.labelsize': 14,  # Axis titles
    'xtick.labelsize': 14,  # X-axis numbers
    'ytick.labelsize': 14   # Y-axis numbers
})

# Richards data points and line function
bins = [0, 20, 40, 60, np.inf]  # Define bins including infinity for the last bin
richards_midpoints = [10, 30, 50, 70]  # Midpoints for each bin
richards_values = [40, 30, 20, 10]  # Hypothetical values from Richards' data
x = np.linspace(0, 100, 1000)
y = -0.5 * x + 45  # Hypothetical line function for Richards

# Plot Richards data and line
plt.plot(richards_midpoints, richards_values, 'bs--', label='Richards Data')
plt.plot(x, y, 'b--', label='Line: y = -0.5x + 45')

# Loop through each city and calculate DBH proportions
for city in cities:
    city_data = df[df['City'] == city]

    # Bin the DBH values and calculate the proportion in each bin
    city_data['Richards_DBH_bins'] = pd.cut(city_data['DBH'], bins=bins)
    Richards_n_classes = city_data['Richards_DBH_bins'].value_counts().sort_index()

    # Calculate the proportion of the total population in each bin
    proportions = Richards_n_classes / Richards_n_classes.sum() * 100  # Convert to percentages

    # Print city name
    print(f"\nCity: {city}")
    print("Midpoint | Count | Proportion (%)")

    # Print the values (counts and proportions) for each bin
    for midpoint, count, proportion in zip(richards_midpoints, Richards_n_classes, proportions):
        print(f"{midpoint:^8} | {count:^5} | {proportion:>10.2f}%")

    # Plot the proportions for this city (only markers, no lines)
    plt.plot(richards_midpoints, proportions, 'o', markersize=4, markerfacecolor='k', markeredgewidth=0,
             label=f'{city} Data')

# Create a line of best fit
x_fit = np.linspace(0, 100, 1000)
y_fit = -0.7119 * x_fit + 53.351  # Hypothetical line of best fit (use appropriate values for your data)
plt.plot(x_fit, y_fit, 'k-', label='Line of Best Fit')

# Add vertical dashed lines at 20 cm, 40 cm, and 60 cm
plt.axvline(x=20, color='gray', linestyle='--')
plt.axvline(x=40, color='gray', linestyle='--')
plt.axvline(x=60, color='gray', linestyle='--')

# Adjust the space above the plot for the labels
plt.subplots_adjust(top=0.85)

# Add text labels for "Young", "Semi-Mature", "Mature", and "Old" using normalized x-coordinates (relative to the axes)
plt.text(0.125, 1.05, 'Young', horizontalalignment='center', transform=plt.gca().transAxes)
plt.text(0.375, 1.05, 'Semi-Mature', horizontalalignment='center', transform=plt.gca().transAxes)
plt.text(0.625, 1.05, 'Mature', horizontalalignment='center', transform=plt.gca().transAxes)
plt.text(0.875, 1.05, 'Old', horizontalalignment='center', transform=plt.gca().transAxes)

# Legend
inventory_data_handle = mlines.Line2D([], [], color='k', marker='o', linestyle='None', markersize=8, label='Inventory Data')
best_fit_handle = mlines.Line2D([], [], color='k', linestyle='-', markersize=8, label='Line of Best Fit')
richards_handle = mlines.Line2D([], [], color='b', marker='s', linestyle='--', markersize=8, label='Richards')
plt.legend(handles=[inventory_data_handle, best_fit_handle, richards_handle])

# Customize plot
plt.xlim(0, 80)
plt.ylim(0, 70)
plt.xlabel('Diameter at Breast Height (cm)')
plt.ylabel('Proportion of Trees (%)')
plt.grid(False)

# Save the figure to file
plt.savefig("Figure 2.png", dpi=900)

plt.show()


# Create a list to hold the data for all cities
output_data = []

# Loop through each city and calculate DBH proportions
for city in cities:
    city_data = df[df['City'] == city]

    # Bin the DBH values and calculate the proportion in each bin
    city_data['Richards_DBH_bins'] = pd.cut(city_data['DBH'], bins=bins)
    Richards_n_classes = city_data['Richards_DBH_bins'].value_counts().sort_index()

    # Calculate the proportion of the total population in each bin
    proportions = Richards_n_classes / Richards_n_classes.sum() * 100  # Convert to percentages

    # Collect the data for this city
    for midpoint, count, proportion in zip(richards_midpoints, Richards_n_classes, proportions):
        output_data.append([city, midpoint, count, proportion])

# Convert the list to a DataFrame
output_df = pd.DataFrame(output_data, columns=['City', 'Midpoint', 'Count', 'Proportion (%)'])

# Save the DataFrame to a CSV file
output_df.to_csv('city_dbh_proportions.csv', index=False)

print("Data saved to 'city_dbh_proportions.csv'")