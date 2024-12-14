import os
import pandas as pd
import re

# All datasets have, in order, Botanical Name, DBH, DAUID, CTUID, and City.

cities = ["Kelowna", "Vancouver", "Victoria", "Calgary", "Edmonton", "Lethbridge", "Strathcona County", "Regina",
          "Winnipeg", "Ajax", "Burlington", "Guelph", "Kingston", "Kitchener", "Mississauga", "Niagara Falls",
          "Ottawa", "St. Catharines", "Toronto", "Waterloo", "Welland", "Whitby", "Windsor", "Longueuil",
          "Montreal", "Quebec City", "Fredericton", "Moncton"]

# Initialize an empty list to hold the data
data_frames = []

# Process each city file
file_path_inventories = r'C:\Users\alexj\Documents\Research\Canadian Urban Forest Inventories - DBH\Python Scripts and Datasets\Inventories'
for city in cities:
    file_name = fr'{file_path_inventories}\{city}.xlsx'

    # Check if the file exists
    if os.path.exists(file_name):
        df = pd.read_excel(file_name)

        # Append the DataFrame to the list
        data_frames.append(df)
    else:
        print(f"{city} file does not exist.")

if len(data_frames) <= 0:
    raise ValueError("No cities XLSX files loaded... Ensure they have been placed in data/cities subdir.")

# Concatenate all DataFrames
master_df = pd.concat(data_frames, ignore_index=True)

# Convert inches to cm in Vancouver
master_df.loc[master_df['City'] == 'Vancouver', 'DBH'] *= 2.54

# Save the master DataFrame to a CSV file
master_df.to_csv(r'C:\Users\alexj\Documents\Research\Canadian Urban Forest Inventories - DBH\Python Scripts and Datasets\(1) Master Dataset.csv', index=False)

print("Merged CSV file created successfully.")

# Load the merged CSV file
merged_df = pd.read_csv(r'C:\Users\alexj\Documents\Research\Canadian Urban Forest Inventories - DBH\Python Scripts and Datasets\(1) Master Dataset.csv', low_memory=False)
species_clean_df = pd.read_csv(r'C:\Users\alexj\Documents\Research\Canadian Urban Forest Inventories - DBH\Python Scripts and Datasets\Non-Inventory Datasets\Find and Replace.csv', low_memory=False)

## Clean the Botanical Name column
initial_count = merged_df.shape[0]

# Deal with blank (missing) species ID, then make all species names lowercase and trim spaces
merged_df['Botanical Name'] = merged_df['Botanical Name'].replace('', pd.NA).fillna('missing')
merged_df['Botanical Name'] = merged_df['Botanical Name'].str.lower().str.strip()
merged_df['Botanical Name'] = merged_df['Botanical Name'].replace('', pd.NA).fillna('missing')

# Standardize cultivars and species
merged_df['Botanical Name'] = merged_df['Botanical Name'].str.replace(" x ", " ", regex=False)
merged_df['Botanical Name'] = merged_df['Botanical Name'].str.replace("'", "", regex=False)

# Use find and replace to deal with spelling mistakes in inventory datasets
for index, row in species_clean_df.iterrows():
    find = row['Find']
    replace = row['Replace']

    # Ensure word boundaries are respected in the replacement to avoid partial replacements
    pattern = r'\b' + re.escape(find) + r'\b'

    merged_df['Botanical Name'] = merged_df['Botanical Name'].str.replace(pattern, replace, regex=True)

# Remove any non-living trees
filtered_df = merged_df[~merged_df["Botanical Name"].isin(["dead", "stump", "stump spp.", "stump for", "shrub", "shrubs", "vine", "vines", "hedge", "vacant"])]
filtered_df = filtered_df[~filtered_df['Botanical Name'].str.contains("vacant", na=False)]

# Add spp. to genus-only identification
def add_spp_if_only_genus(name):
    if isinstance(name, str) and len(name.strip().split()) == 1:  # Trim spaces and check if there's only one word
        return name.strip() + " spp."  # Trim spaces and add spp.
    return name.strip()  # Trim spaces in any case
filtered_df.loc[:, 'Botanical Name'] = filtered_df['Botanical Name'].apply(add_spp_if_only_genus) # Apply the function to the 'Botanical Name' column
final_count = filtered_df.shape[0]

# Remove any incorrect letters
filtered_df['Botanical Name'] = filtered_df['Botanical Name'].str.replace("stump spp.", "missing", regex=False)
filtered_df['Botanical Name'] = filtered_df['Botanical Name'].str.replace("missing.", "missing", regex=False)
filtered_df['Botanical Name'] = filtered_df['Botanical Name'].str.replace("missing spp.", "missing", regex=False)
filtered_df['Botanical Name'] = filtered_df['Botanical Name'].str.replace("spp. spp.", "missing", regex=False)

## Clean the DBH column
filtered_df.loc[:, "DBH"] = pd.to_numeric(filtered_df["DBH"], errors='coerce')
filtered_df.loc[filtered_df["DBH"] > 350, "DBH"] = 0
filtered_df.loc[filtered_df["DBH"] == 0, "DBH"] = pd.NA
num_rows_with_dbh_0 = filtered_df[filtered_df["DBH"] == 0].shape[0]
print(f"Number of rows with DBH of 0: {num_rows_with_dbh_0}")

# Calculate basal area
filtered_df.loc[:, 'Basal Area'] = 0.00007854 * (filtered_df['DBH'] ** 2)

# Remove rows where both DAUID and CTUID are missing
filtered_df = filtered_df[~(filtered_df["DAUID"].isna() & filtered_df["CTUID"].isna())]

# Count the number of rows after removing rows with missing DAUID and CTUID
final_count_after_missing_removal = filtered_df.shape[0]

# Find and count unique values of DAUID where CTUID is blank
blank_ctuid_df = filtered_df[filtered_df["CTUID"].isna()]
unique_dauid_with_blank_ctuid = blank_ctuid_df["DAUID"].unique()
num_instances_blank_ctuid = blank_ctuid_df.shape[0]

print("Unique DAUID values where CTUID is blank:")
print(unique_dauid_with_blank_ctuid)
print(f"Number of instances where CTUID is blank: {num_instances_blank_ctuid}")

# Fill missing CTUID values
data_dict = {
    'DAUID': [59150883, 59150891, 59153562, 59170272, 35240431, 35100286, 35201466, 35204675, 35204821, 35205067,
              35370553, 24580007, 24662985, 24662707, 24662951, 24662821, 24662885, 24660984, 24663395, 24230066,
              13100304, 13070131, 12090576, 59154073],
    'CTUID': [9330045.01, 9330025, 9330059.08, 9350001, 5370204, 5210100.01, 5350003, 5350210.04, 5350012.01, 5350200.01,
              5590043.02, 4620886.03, 4620288, 4620276, 4620585.01, 4620290.05, 4620290.09, 4620322.03, 4620390, 4210310,
              3200009, 3050006, 2050112, 9330202.01],
    'City': ['Vancouver', 'Vancouver', 'Vancouver', 'Victoria', 'Burlington', 'Kingston', 'Toronto', 'Toronto', 'Toronto', 'Toronto',
             'Windsor', 'Longueuil', 'Montreal', 'Montreal', 'Montreal', 'Montreal', 'Montreal', 'Montreal', 'Montreal', 'Quebec City',
             'Fredericton', 'Moncton', 'Halifax', 'New Westminster']
}

data_dict_df = pd.DataFrame(data_dict)

# Merge the filtered DataFrame with the data dictionary DataFrame based on "DAUID"
filtered_df = filtered_df.merge(data_dict_df, on='DAUID', how='left', suffixes=('', '_dict'))

# Fill missing CTUID and City values from the dictionary
filtered_df['CTUID'] = filtered_df['CTUID'].combine_first(filtered_df['CTUID_dict'])
filtered_df['City'] = filtered_df['City'].combine_first(filtered_df['City_dict'])

# Drop the extra columns from the dictionary
filtered_df = filtered_df.drop(columns=['CTUID_dict', 'City_dict'])
print("CTUID and City columns updated based on DAUID.")

# Count the number of instances of DAUID with blank CTUID after filling
blank_ctuid_df_after_filling = filtered_df[filtered_df["CTUID"].isna()]
num_instances_blank_ctuid_after_filling = blank_ctuid_df_after_filling.shape[0]

# Save the updated DataFrame to the master CSV file
filtered_df.to_csv(r'C:\Users\alexj\Documents\Research\Canadian Urban Forest Inventories - DBH\Python Scripts and Datasets\(2) Filtered Master Dataset.csv', index=False)

# Print the counts
print(f"Number of trees where CTUID is blank after filling: {num_instances_blank_ctuid_after_filling}")
print(f"Number of trees after removing trees with missing DAUID and CTUID: {final_count_after_missing_removal}")
print(f"Number of out-of-city trees removed: {final_count - final_count_after_missing_removal}")
print(f"Number of dead trees, stumps, missing, etc. removed: {initial_count - final_count}")

print("Filtered Master Dataset file created successfully.")
