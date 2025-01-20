"""
تنفيذ خوارزمية K-means للتجميع
"""
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

class KMeansAnalysis:
    def __init__(self, data):
        self.data = data
        self.model = None
        
    def prepare_data(self):
        """تجهيز البيانات للتجميع"""
        pass
        
    def find_optimal_clusters(self, max_clusters=10):
        """تحديد العدد الأمثل للمجموعات"""
        silhouette_scores = []
        for k in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            cluster_labels = kmeans.fit_predict(self.data)
            silhouette_avg = silhouette_score(self.data, cluster_labels)
            silhouette_scores.append(silhouette_avg)
        return silhouette_scores
        
    def train_model(self, n_clusters):
        """تدريب نموذج K-means"""
        self.model = KMeans(n_clusters=n_clusters, random_state=42)
        self.model.fit(self.data)
        
    def get_clusters(self):
        """الحصول على تصنيفات المجموعات"""
        return self.model.labels_
