import os
import pandas as pd

# All datasets have, in order, Botanical Name, DBH, DAUID, CTUID, and City.

cities = ["Kelowna", "Maple Ridge", "New Westminster", "Vancouver", "Victoria", "Calgary", "Edmonton", "Lethbridge",
          "Strathcona County", "Regina", "Winnipeg", "Ajax", "Burlington", "Guelph", "Kingston", "Kitchener",
          "Mississauga", "Niagara Falls", "Ottawa", "Peterborough", "St. Catharines", "Toronto", "Waterloo", "Welland",
          "Whitby", "Windsor", "Longueuil", "Montreal", "Quebec City", "Fredericton", "Moncton", "Halifax"]

# Initialize an empty list to hold the data
data_frames = []

# Process each city file
file_path_inventories = r'C:\Users\alexj\Documents\Research\Canadian Urban Forest Inventories - Structure and Diversity\Python Scripts and Datasets\Inventories'
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

## Species codes to scientific binomials
# Load the data dictionaries
file_path_species_codes = r'C:\Users\alexj\Documents\Research\Canadian Urban Forest Inventories - Structure and Diversity\Python Scripts and Datasets\Non-Inventory Datasets\Tree Codes'
Halifax_dict = pd.read_csv(fr'{file_path_species_codes}\Halifax.csv')
Mississauga_dict = pd.read_csv(fr'{file_path_species_codes}\Mississauga.csv')
Moncton_dict = pd.read_csv(fr'{file_path_species_codes}\Moncton.csv')
Ottawa_dict = pd.read_csv(fr'{file_path_species_codes}\Ottawa.csv')
Toronto_dict = pd.read_csv(fr'{file_path_species_codes}\Toronto.csv')

# Replace the species codes
def replace_botanical_name(row):
    if row['City'] == 'Halifax':
        code_dict = Halifax_dict
    elif row['City'] == 'Mississauga':
        code_dict = Mississauga_dict
    elif row['City'] == 'Moncton':
        code_dict = Moncton_dict
    elif row['City'] == 'Ottawa':
        code_dict = Ottawa_dict
    elif row['City'] == 'Toronto':
        code_dict = Toronto_dict
    else:
        return row['Botanical Name']  # If city doesn't match, return the original Botanical Name

    # Try to match the code and return the corresponding botanical name
    match = code_dict[code_dict['Code'] == row['Botanical Name']]
    if not match.empty:
        return match['Botanical Name'].values[0]
    else:
        return row['Botanical Name']  # If no match is found, keep the original value

master_df['Botanical Name'] = master_df.apply(replace_botanical_name, axis=1) # Apply the function to the DataFrame

# Save the master DataFrame to a CSV file
master_df.to_csv(r'C:\Users\alexj\Documents\Research\Canadian Urban Forest Inventories - Structure and Diversity\Python Scripts and Datasets\(1) Master Dataset.csv', index=False)

print("Merged CSV file created successfully.")