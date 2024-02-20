import pandas as pd

# Replace 'your_file.csv' with the actual path to your CSV file
file_path = '.data/Image_Collections/ADNI1_Annual_2_Yr_3T_2_20_2024.csv'

# Read the CSV file into a pandas DataFrame
df = pd.read_csv(file_path)

# Group by 'Subject' and aggregate 'Description' column
grouped_data = df.groupby('Subject')['Description'].apply(list).reset_index()

# Save the grouped data to a new CSV file
grouped_data.to_csv('grouped_file.csv', index=False)