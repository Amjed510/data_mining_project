import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import os
import sys
import traceback

def print_message(msg):
    """Print message and force flush"""
    print(msg)
    sys.stdout.flush()

def load_data():
    """Load and prepare test data"""
    try:
        print_message("Loading processed data...")
        data_path = os.path.join(os.getcwd(), 'data', 'processed_data.csv')
        print_message(f"Loading data from: {data_path}")
        df = pd.read_csv(data_path)
        print_message(f"Data loaded successfully. Shape: {df.shape}")
        
        # Prepare features
        features = ['Category', 'Sub-Category', 'Sales', 'Ship Mode', 'Region',
                   'Segment', 'Year', 'Month', 'ShippingDays']
        df_selected = df[features]
        
        # Convert categorical variables
        categorical_cols = ['Category', 'Sub-Category', 'Segment', 'Region', 'Ship Mode']
        df_encoded = pd.get_dummies(df_selected, columns=categorical_cols)
        
        # Create target variable
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

def load_models():
    """Load trained models"""
    try:
        print_message("\nLoading trained models...")
        models = {}
        model_files = {
            'decision_tree': 'models/decision_tree_model.pkl',
            'naive_bayes': 'models/naive_bayes_model.pkl',
            'kmeans': 'models/kmeans_model.pkl',
            'scaler': 'models/scaler.pkl'
        }
        
        for name, file in model_files.items():
            model_path = os.path.join(os.getcwd(), file)
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    models[name] = pickle.load(f)
                print_message(f"Loaded {name} model from {model_path}")
            else:
                print_message(f"Warning: {model_path} not found")
        
        return models
        
    except Exception as e:
        print_message("Error loading models:")
        print_message(str(e))
        print_message("\nFull error details:")
        print_message(traceback.format_exc())
        return None

def test_classification_models(models, X_test, y_test):
    """Test classification models"""
    try:
        print_message("\nTesting classification models...")
        results = {}
        
        # Test Decision Tree
        if 'decision_tree' in models:
            print_message("\nDecision Tree Results:")
            dt_pred = models['decision_tree'].predict(X_test)
            dt_accuracy = accuracy_score(y_test, dt_pred)
            dt_report = classification_report(y_test, dt_pred)
            
            results['decision_tree'] = {
                'accuracy': dt_accuracy,
                'report': dt_report
            }
            
            print_message(f"Accuracy: {dt_accuracy:.4f}")
            print_message("\nClassification Report:")
            print_message(dt_report)
        
        # Test Naive Bayes
        if 'naive_bayes' in models:
            print_message("\nNaive Bayes Results:")
            nb_pred = models['naive_bayes'].predict(X_test)
            nb_accuracy = accuracy_score(y_test, nb_pred)
            nb_report = classification_report(y_test, nb_pred)
            
            results['naive_bayes'] = {
                'accuracy': nb_accuracy,
                'report': nb_report
            }
            
            print_message(f"Accuracy: {nb_accuracy:.4f}")
            print_message("\nClassification Report:")
            print_message(nb_report)
        
        return results
            
    except Exception as e:
        print_message("Error testing classification models:")
        print_message(str(e))
        print_message("\nFull error details:")
        print_message(traceback.format_exc())
        return None

def test_clustering_model(models, X_cluster):
    """Test clustering model"""
    try:
        print_message("\nTesting clustering model...")
        
        if 'scaler' in models and 'kmeans' in models:
            # Scale the data
            X_scaled = models['scaler'].transform(X_cluster)
            
            # Predict clusters
            clusters = models['kmeans'].predict(X_scaled)
            
            # Analyze cluster statistics
            X_cluster_with_labels = X_cluster.copy()
            X_cluster_with_labels['Cluster'] = clusters
            
            print_message("\nCluster Statistics:")
            cluster_stats = X_cluster_with_labels.groupby('Cluster').agg({
                'Sales': ['mean', 'std', 'count'],
                'Year': 'mean',
                'Month': 'mean',
                'ShippingDays': 'mean'
            })
            print_message("\nMean values for each cluster:")
            print_message(str(cluster_stats))
            
            return clusters
            
    except Exception as e:
        print_message("Error testing clustering model:")
        print_message(str(e))
        print_message("\nFull error details:")
        print_message(traceback.format_exc())
        return None

def save_results(clusters=None, classification_results=None):
    """Save test results"""
    try:
        print_message("\nSaving test results...")
        
        # Create results directory if it doesn't exist
        results_dir = os.path.join(os.getcwd(), 'results')
        os.makedirs(results_dir, exist_ok=True)
        
        # Save classification results
        if classification_results is not None:
            eval_path = os.path.join(results_dir, 'model_evaluation.txt')
            with open(eval_path, 'w', encoding='utf-8') as f:
                f.write("# Model Evaluation Results\n\n")
                
                # Write Decision Tree results
                if 'decision_tree' in classification_results:
                    f.write("## Decision Tree Results\n")
                    f.write(f"Accuracy: {classification_results['decision_tree']['accuracy']:.4f}\n")
                    f.write("\nClassification Report:\n")
                    f.write(classification_results['decision_tree']['report'])
                    f.write("\n")
                
                # Write Naive Bayes results
                if 'naive_bayes' in classification_results:
                    f.write("\n## Naive Bayes Results\n")
                    f.write(f"Accuracy: {classification_results['naive_bayes']['accuracy']:.4f}\n")
                    f.write("\nClassification Report:\n")
                    f.write(classification_results['naive_bayes']['report'])
                    f.write("\n")
            
            print_message(f"Saved model evaluation results to {eval_path}")
        
        # Save cluster assignments if available
        if clusters is not None:
            cluster_path = os.path.join(results_dir, 'cluster_assignments.csv')
            cluster_df = pd.DataFrame({'Cluster': clusters})
            cluster_df.to_csv(cluster_path, index=False)
            print_message(f"Saved cluster assignments to {cluster_path}")
            
    except Exception as e:
        print_message("Error saving results:")
        print_message(str(e))
        print_message("\nFull error details:")
        print_message(traceback.format_exc())

def main():
    try:
        # Load data
        print_message("Step 1: Loading and preparing data...")
        X_classification, y_classification, X_clustering = load_data()
        if X_classification is None:
            print_message("Failed to load data. Exiting...")
            return
            
        # Load models
        print_message("\nStep 2: Loading models...")
        models = load_models()
        if models is None:
            print_message("Failed to load models. Exiting...")
            return
            
        # Test classification models
        print_message("\nStep 3: Testing classification models...")
        classification_results = test_classification_models(models, X_classification, y_classification)
        
        # Test clustering model
        print_message("\nStep 4: Testing clustering model...")
        clusters = test_clustering_model(models, X_clustering)
        
        # Save results
        save_results(clusters, classification_results)
        
        print_message("\nTesting complete! Check the 'results' directory for detailed results.")
        
    except Exception as e:
        print_message("Error in main execution:")
        print_message(str(e))
        print_message("\nFull error details:")
        print_message(traceback.format_exc())

if __name__ == "__main__":
    main()
