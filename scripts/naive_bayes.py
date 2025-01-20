"""
تنفيذ خوارزمية Naive Bayes
"""
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

class NaiveBayesAnalysis:
    def __init__(self, data):
        self.data = data
        self.model = None
        
    def prepare_data(self):
        """تجهيز البيانات للتدريب"""
        pass
        
    def train_model(self, X, y):
        """تدريب نموذج Naive Bayes"""
        self.model = GaussianNB()
        self.model.fit(X, y)
        
    def evaluate_model(self, X_test, y_test):
        """تقييم أداء النموذج"""
        predictions = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions)
        return accuracy, report
