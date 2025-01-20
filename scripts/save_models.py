import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pickle
import os
import sys
import traceback

def print_message(msg):
    """Print message and force flush"""
    print(msg)
    sys.stdout.flush()

def load_and_prepare_data():
    """Load and prepare data for training"""
    try:
        print_message("Loading processed data...")
        data_path = os.path.join(os.getcwd(), 'data', 'processed_data.csv')
        print_message(f"Loading data from: {data_path}")
        df = pd.read_csv(data_path)
        print_message(f"Data loaded successfully. Shape: {df.shape}")
        
        # Prepare features
        print_message("\nPreparing features...")
        features = ['Category', 'Sub-Category', 'Sales', 'Ship Mode', 'Region',
                   'Segment', 'Year', 'Month', 'ShippingDays']
        df_selected = df[features]
        
        # Convert categorical variables
        print_message("Converting categorical variables...")
        categorical_cols = ['Category', 'Sub-Category', 'Segment', 'Region', 'Ship Mode']
        df_encoded = pd.get_dummies(df_selected, columns=categorical_cols)
        
        # Create target variable for classification (high/low profit)
        df_encoded['Sales_Category'] = (df_encoded['Sales'] > df_encoded['Sales'].median()).astype(int)
        
        # Prepare features for different models
        X_classification = df_encoded.drop(['Sales', 'Sales_Category'], axis=1)
        y_classification = df_encoded['Sales_Category']
        
        X_clustering = df[['Sales', 'Year', 'Month', 'ShippingDays']]
        
        print_message("Features prepared successfully.")
        print_message(f"Classification features shape: {X_classification.shape}")
        print_message(f"Clustering features shape: {X_clustering.shape}")
        
        return X_classification, y_classification, X_clustering
        
    except Exception as e:
        print_message("Error loading data:")
        print_message(str(e))
        print_message("\nFull error details:")
        print_message(traceback.format_exc())
        return None, None, None

def train_and_save_models(X_classification, y_classification, X_clustering):
    """Train and save models"""
    try:
        print_message("\nTraining and saving models...")
        current_dir = os.getcwd()
        print_message(f"Current working directory: {current_dir}")
        
        # Create models directory if it doesn't exist
        models_dir = os.path.abspath(os.path.join(current_dir, 'models'))
        os.makedirs(models_dir, exist_ok=True)
        print_message(f"Models will be saved to: {models_dir}")
        
        try:
            # Train and save Decision Tree
            print_message("\nTraining Decision Tree...")
            dt_model = DecisionTreeClassifier(random_state=42)
            dt_model.fit(X_classification, y_classification)
            dt_path = os.path.join(models_dir, 'decision_tree_model.pkl')
            print_message(f"Attempting to save Decision Tree model to {dt_path}")
            with open(dt_path, 'wb') as f:
                pickle.dump(dt_model, f)
            print_message(f"Successfully saved Decision Tree model to {dt_path}")
        except Exception as e:
            print_message(f"Error saving Decision Tree model: {str(e)}")
            print_message(traceback.format_exc())
        
        try:
            # Train and save Naive Bayes
            print_message("\nTraining Naive Bayes...")
            nb_model = GaussianNB()
            nb_model.fit(X_classification, y_classification)
            nb_path = os.path.join(models_dir, 'naive_bayes_model.pkl')
            print_message(f"Attempting to save Naive Bayes model to {nb_path}")
            with open(nb_path, 'wb') as f:
                pickle.dump(nb_model, f)
            print_message(f"Successfully saved Naive Bayes model to {nb_path}")
        except Exception as e:
            print_message(f"Error saving Naive Bayes model: {str(e)}")
            print_message(traceback.format_exc())
        
        try:
            # Train and save K-means
            print_message("\nTraining K-means...")
            scaler = StandardScaler()
            X_clustering_scaled = scaler.fit_transform(X_clustering)
            
            kmeans = KMeans(n_clusters=4, random_state=42)
            kmeans.fit(X_clustering_scaled)
            
            kmeans_path = os.path.join(models_dir, 'kmeans_model.pkl')
            print_message(f"Attempting to save K-means model to {kmeans_path}")
            with open(kmeans_path, 'wb') as f:
                pickle.dump(kmeans, f)
            print_message(f"Successfully saved K-means model to {kmeans_path}")
            
            scaler_path = os.path.join(models_dir, 'scaler.pkl')
            print_message(f"Attempting to save scaler to {scaler_path}")
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
            print_message(f"Successfully saved scaler to {scaler_path}")
        except Exception as e:
            print_message(f"Error saving K-means model: {str(e)}")
            print_message(traceback.format_exc())
            
    except Exception as e:
        print_message("Error in train_and_save_models:")
        print_message(str(e))
        print_message("\nFull error details:")
        print_message(traceback.format_exc())

def main():
    try:
        # Load and prepare data
        print_message("Step 1: Loading and preparing data...")
        X_classification, y_classification, X_clustering = load_and_prepare_data()
        if X_classification is None:
            print_message("Failed to load data. Exiting...")
            return
            
        # Train and save models
        print_message("\nStep 2: Training and saving models...")
        train_and_save_models(X_classification, y_classification, X_clustering)
        
        print_message("\nAll models have been trained and saved!")
        
    except Exception as e:
        print_message("Error in main execution:")
        print_message(str(e))
        print_message("\nFull error details:")
        print_message(traceback.format_exc())

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding='utf-8')
    main()
