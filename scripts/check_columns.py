import pandas as pd

# Load the data
df = pd.read_csv('data/processed_data.csv')

# Print column names
print("Column names:")
for col in df.columns:
    print(f"- {col}")
