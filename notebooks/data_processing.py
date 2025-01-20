import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import traceback
import os
import sys

# تعيين ترميز المخرجات
sys.stdout.reconfigure(encoding='utf-8')

def main():
    try:
        print("Current working directory:", os.getcwd(), flush=True)
        
        # Load data
        print("Loading data from train.csv...", flush=True)
        data_path = 'data/train.csv'
        print(f"Attempting to load data from: {os.path.abspath(data_path)}", flush=True)
        df = pd.read_csv(data_path)
        print(f"Data loaded successfully. Shape: {df.shape}", flush=True)
        
        # Display initial information
        print("\nData information:", flush=True)
        df.info()
        
        # Handle missing values
        print("\nHandling missing values...", flush=True)
        missing_before = df.isnull().sum()
        print("Missing values before processing:", flush=True)
        print(missing_before, flush=True)
        df = df.dropna()
        missing_after = df.isnull().sum()
        print("\nMissing values after processing:", flush=True)
        print(missing_after, flush=True)
        print(f"Rows before: {len(df) + missing_before.sum()}", flush=True)
        print(f"Rows after: {len(df)}", flush=True)
        
        # Handle outliers
        print("\nHandling outliers...", flush=True)
        negative_sales = len(df[df['Sales'] < 0])
        print(f"Number of negative sales: {negative_sales}", flush=True)
        df = df[df['Sales'] >= 0]
        print(f"Removed {negative_sales} rows with negative sales", flush=True)
        
        # Add features
        print("\nAdding new features...", flush=True)
        
        # Convert dates with correct format
        df['Order Date'] = pd.to_datetime(df['Order Date'], format='%d/%m/%Y')
        df['Ship Date'] = pd.to_datetime(df['Ship Date'], format='%d/%m/%Y')
        
        # Extract date features
        df['Year'] = df['Order Date'].dt.year
        df['Month'] = df['Order Date'].dt.month
        df['Day'] = df['Order Date'].dt.day
        df['DayOfWeek'] = df['Order Date'].dt.dayofweek
        df['ShippingDays'] = (df['Ship Date'] - df['Order Date']).dt.days
        
        print("Added features:", ['Year', 'Month', 'Day', 'DayOfWeek', 'ShippingDays'], flush=True)
        
        # Save processed data
        try:
            output_path = os.path.join('data', 'processed_data.csv')
            os.makedirs('data', exist_ok=True)
            print(f"Attempting to save data to: {output_path}", flush=True)
            df.to_csv(output_path, index=False, encoding='utf-8')
            print(f"\nData saved to {output_path}", flush=True)
            print(f"Final shape: {df.shape}", flush=True)
        except Exception as save_error:
            print("Error saving data:", flush=True)
            print(str(save_error), flush=True)
            print("\nFull error details:", flush=True)
            print(traceback.format_exc(), flush=True)
        
        return df
        
    except Exception as e:
        print("Error in main processing:", flush=True)
        print(str(e), flush=True)
        print("\nFull error details:", flush=True)
        print(traceback.format_exc(), flush=True)
        return None

if __name__ == "__main__":
    print("Starting program execution...", flush=True)
    df = main()
    print("Program execution completed successfully!", flush=True)
