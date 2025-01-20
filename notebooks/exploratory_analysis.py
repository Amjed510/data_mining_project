import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import sys
import traceback

# Set style for plots
plt.style.use('seaborn')
sns.set_palette('husl')

def load_data():
    """Load the processed data"""
    try:
        print("Loading processed data...", flush=True)
        df = pd.read_csv('data/processed_data.csv')
        
        # Convert dates back to datetime
        df['Order Date'] = pd.to_datetime(df['Order Date'])
        df['Ship Date'] = pd.to_datetime(df['Ship Date'])
        
        print(f"Data loaded successfully. Shape: {df.shape}", flush=True)
        return df
        
    except Exception as e:
        print("Error loading data:", flush=True)
        print(str(e), flush=True)
        print("\nFull error details:", flush=True)
        print(traceback.format_exc(), flush=True)
        return None

def analyze_sales_trends(df):
    """Analyze sales trends over time"""
    try:
        print("\nAnalyzing sales trends...", flush=True)
        
        # Monthly sales trend
        plt.figure(figsize=(12, 6))
        monthly_sales = df.groupby(['Year', 'Month'])['Sales'].sum().reset_index()
        monthly_sales['Date'] = pd.to_datetime(monthly_sales[['Year', 'Month']].assign(DAY=1))
        plt.plot(monthly_sales['Date'], monthly_sales['Sales'])
        plt.title('Monthly Sales Trend')
        plt.xlabel('Date')
        plt.ylabel('Total Sales')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Create directory if it doesn't exist
        os.makedirs('results/visualizations', exist_ok=True)
        plt.savefig('results/visualizations/monthly_sales_trend.png')
        plt.close()
        
        print("Sales trends analysis completed.", flush=True)
        
    except Exception as e:
        print("Error analyzing sales trends:", flush=True)
        print(str(e), flush=True)
        print("\nFull error details:", flush=True)
        print(traceback.format_exc(), flush=True)

def analyze_category_distribution(df):
    """Analyze product category distribution"""
    try:
        print("\nAnalyzing category distribution...", flush=True)
        
        # Category distribution
        plt.figure(figsize=(10, 6))
        category_sales = df.groupby('Category')['Sales'].sum().sort_values(ascending=False)
        category_sales.plot(kind='bar')
        plt.title('Sales by Category')
        plt.xlabel('Category')
        plt.ylabel('Total Sales')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('results/visualizations/sales_by_category.png')
        plt.close()
        
        print("Category distribution analysis completed.", flush=True)
        
    except Exception as e:
        print("Error analyzing category distribution:", flush=True)
        print(str(e), flush=True)
        print("\nFull error details:", flush=True)
        print(traceback.format_exc(), flush=True)

def analyze_customer_segments(df):
    """Analyze customer segments"""
    try:
        print("\nAnalyzing customer segments...", flush=True)
        
        # Sales by segment
        plt.figure(figsize=(8, 6))
        segment_sales = df.groupby('Segment')['Sales'].sum()
        plt.pie(segment_sales, labels=segment_sales.index, autopct='%1.1f%%')
        plt.title('Sales Distribution by Customer Segment')
        plt.axis('equal')
        plt.savefig('results/visualizations/sales_by_segment.png')
        plt.close()
        
        print("Customer segments analysis completed.", flush=True)
        
    except Exception as e:
        print("Error analyzing customer segments:", flush=True)
        print(str(e), flush=True)
        print("\nFull error details:", flush=True)
        print(traceback.format_exc(), flush=True)

def generate_summary_statistics(df):
    """Generate summary statistics"""
    try:
        print("\nGenerating summary statistics...", flush=True)
        
        # Sales statistics
        sales_stats = df['Sales'].describe()
        print("\nSales Statistics:", flush=True)
        print(sales_stats, flush=True)
        
        # Save statistics
        os.makedirs('results', exist_ok=True)
        with open('results/summary_statistics.txt', 'w', encoding='utf-8') as f:
            f.write("Sales Statistics:\n")
            f.write(str(sales_stats))
            f.write("\n\nTop 5 Categories by Sales:\n")
            top_categories = df.groupby('Category')['Sales'].sum().sort_values(ascending=False).head()
            f.write(str(top_categories))
        
        print("Summary statistics generated.", flush=True)
        
    except Exception as e:
        print("Error generating summary statistics:", flush=True)
        print(str(e), flush=True)
        print("\nFull error details:", flush=True)
        print(traceback.format_exc(), flush=True)

def main():
    try:
        # Load data
        df = load_data()
        if df is None:
            print("Failed to load data. Exiting...", flush=True)
            return
        
        # Create directories
        print("\nCreating output directories...", flush=True)
        os.makedirs('results/visualizations', exist_ok=True)
        
        # Perform analysis
        analyze_sales_trends(df)
        analyze_category_distribution(df)
        analyze_customer_segments(df)
        generate_summary_statistics(df)
        
        print("\nAnalysis complete! Check the 'results' directory for visualizations and statistics.", flush=True)
        
    except Exception as e:
        print("Error in main execution:", flush=True)
        print(str(e), flush=True)
        print("\nFull error details:", flush=True)
        print(traceback.format_exc(), flush=True)

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding='utf-8')
    main()
