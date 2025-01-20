import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import os
import sys
import traceback

def load_data():
    """Load the processed data"""
    try:
        print("Loading processed data...", flush=True)
        df = pd.read_csv('data/processed_data.csv')
        print(f"Data loaded successfully. Shape: {df.shape}", flush=True)
        return df
    except Exception as e:
        print("Error loading data:", flush=True)
        print(str(e), flush=True)
        return None

def prepare_features(df):
    """Prepare features for sales prediction"""
    try:
        print("\nPreparing features...", flush=True)
        
        # Convert dates to datetime
        df['Order Date'] = pd.to_datetime(df['Order Date'])
        
        # Create features
        features = pd.DataFrame()
        features['Year'] = df['Order Date'].dt.year
        features['Month'] = df['Order Date'].dt.month
        features['Category'] = pd.get_dummies(df['Category'], prefix='Category')
        features['Segment'] = pd.get_dummies(df['Segment'], prefix='Segment')
        features['Region'] = pd.get_dummies(df['Region'], prefix='Region')
        
        # Target variable
        target = df['Sales']
        
        print("Features prepared successfully.", flush=True)
        return features, target
        
    except Exception as e:
        print("Error preparing features:", flush=True)
        print(str(e), flush=True)
        return None, None

def train_sales_prediction_model(features, target):
    """Train a Random Forest model for sales prediction"""
    try:
        print("\nTraining sales prediction model...", flush=True)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=0.2, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print("\nModel Performance:", flush=True)
        print(f"Mean Squared Error: {mse:.2f}", flush=True)
        print(f"R2 Score: {r2:.2f}", flush=True)
        
        # Save metrics
        with open('results/model_evaluation.txt', 'w') as f:
            f.write(f"Sales Prediction Model Performance:\n")
            f.write(f"Mean Squared Error: {mse:.2f}\n")
            f.write(f"R2 Score: {r2:.2f}\n")
        
        print("Model training completed.", flush=True)
        return model, scaler
        
    except Exception as e:
        print("Error training model:", flush=True)
        print(str(e), flush=True)
        return None, None

def perform_market_basket_analysis(df):
    """Perform market basket analysis using Apriori algorithm"""
    try:
        print("\nPerforming market basket analysis...", flush=True)
        
        # Prepare data for market basket analysis
        basket = pd.crosstab(df['Order ID'], df['Sub-Category'])
        basket = (basket > 0).astype(int)
        
        # Generate frequent itemsets
        frequent_itemsets = apriori(basket, min_support=0.01, use_colnames=True)
        
        # Generate association rules
        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
        
        # Sort rules by lift
        rules = rules.sort_values('lift', ascending=False)
        
        # Save results
        rules.to_csv('results/association_rules.csv', index=False)
        print(f"Found {len(rules)} association rules.", flush=True)
        
        # Save top 10 rules to text file
        with open('results/market_basket_analysis.txt', 'w') as f:
            f.write("Top 10 Product Association Rules:\n\n")
            for idx, row in rules.head(10).iterrows():
                f.write(f"Rule {idx+1}:\n")
                f.write(f"Products: {row['antecedents']} -> {row['consequents']}\n")
                f.write(f"Support: {row['support']:.3f}\n")
                f.write(f"Confidence: {row['confidence']:.3f}\n")
                f.write(f"Lift: {row['lift']:.3f}\n\n")
        
        print("Market basket analysis completed.", flush=True)
        return rules
        
    except Exception as e:
        print("Error in market basket analysis:", flush=True)
        print(str(e), flush=True)
        return None

def main():
    try:
        # Create results directory
        os.makedirs('results', exist_ok=True)
        
        # Load data
        df = load_data()
        if df is None:
            print("Failed to load data. Exiting...", flush=True)
            return
            
        # Sales Prediction
        print("\n=== Sales Prediction ===", flush=True)
        features, target = prepare_features(df)
        if features is not None and target is not None:
            model, scaler = train_sales_prediction_model(features, target)
        
        # Market Basket Analysis
        print("\n=== Market Basket Analysis ===", flush=True)
        rules = perform_market_basket_analysis(df)
        
        print("\nAnalysis complete! Check the 'results' directory for detailed results.", flush=True)
        
    except Exception as e:
        print("Error in main execution:", flush=True)
        print(str(e), flush=True)
        print("\nFull error details:", flush=True)
        print(traceback.format_exc(), flush=True)

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding='utf-8')
    main()
