"""
تطبيق خوارزمية K-means للتجميع
"""
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
from datetime import datetime

# إعداد التسجيل
logging.basicConfig(
    filename='data/kmeans_analysis.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_data():
    """تحميل البيانات"""
    try:
        df = pd.read_csv('data/raw/sales_data.csv')
        logging.info(f"تم تحميل {len(df)} سجل")
        return df
    except Exception as e:
        logging.error(f"خطأ في تحميل البيانات: {str(e)}")
        return None

def prepare_data(df):
    """تحضير البيانات للتجميع"""
    try:
        # معالجة القيم المفقودة
        numeric_cols = ['Sales', 'Quantity', 'Discount', 'Profit']
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].median())
        
        # تطبيع البيانات
        scaler = StandardScaler()
        df_scaled = pd.DataFrame(
            scaler.fit_transform(df[numeric_cols]),
            columns=numeric_cols
        )
        
        logging.info("تم تحضير البيانات بنجاح")
        return df_scaled, df[numeric_cols]
    except Exception as e:
        logging.error(f"خطأ في تحضير البيانات: {str(e)}")
        return None

def find_optimal_clusters(data):
    """تحديد العدد الأمثل للمجموعات"""
    try:
        n_clusters_range = range(2, 11)
        silhouette_scores = []
        inertias = []
        
        for n_clusters in n_clusters_range:
            # تدريب النموذج
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            kmeans.fit(data)
            
            # حساب معامل سيلويت
            silhouette_avg = silhouette_score(data, kmeans.labels_)
            silhouette_scores.append(silhouette_avg)
            
            # حساب مجموع المربعات داخل المجموعات
            inertias.append(kmeans.inertia_)
        
        # رسم منحنى Elbow
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(n_clusters_range, inertias, marker='o')
        plt.xlabel('عدد المجموعات')
        plt.ylabel('مجموع المربعات داخل المجموعات')
        plt.title('منحنى Elbow')
        
        # رسم معاملات سيلويت
        plt.subplot(1, 2, 2)
        plt.plot(n_clusters_range, silhouette_scores, marker='o')
        plt.xlabel('عدد المجموعات')
        plt.ylabel('معامل سيلويت')
        plt.title('معاملات سيلويت')
        
        plt.tight_layout()
        plt.savefig('results/kmeans_optimal_clusters.png')
        plt.close()
        
        # اختيار العدد الأمثل للمجموعات
        optimal_clusters = n_clusters_range[np.argmax(silhouette_scores)]
        
        logging.info(f"تم تحديد العدد الأمثل للمجموعات: {optimal_clusters}")
        return optimal_clusters
    except Exception as e:
        logging.error(f"خطأ في تحديد العدد الأمثل للمجموعات: {str(e)}")
        return None

def perform_clustering(data, n_clusters):
    """تنفيذ التجميع"""
    try:
        # تدريب النموذج
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(data)
        
        logging.info("تم تنفيذ التجميع بنجاح")
        return clusters, kmeans.cluster_centers_
    except Exception as e:
        logging.error(f"خطأ في تنفيذ التجميع: {str(e)}")
        return None

def analyze_clusters(data, clusters, centers, original_data):
    """تحليل المجموعات"""
    try:
        # إضافة المجموعات إلى البيانات الأصلية
        clustered_data = original_data.copy()
        clustered_data['Cluster'] = clusters
        
        # حساب إحصائيات المجموعات
        cluster_stats = clustered_data.groupby('Cluster').agg({
            'Sales': ['mean', 'std', 'min', 'max'],
            'Quantity': ['mean', 'std', 'min', 'max'],
            'Discount': ['mean', 'std', 'min', 'max'],
            'Profit': ['mean', 'std', 'min', 'max']
        })
        
        # رسم توزيع المجموعات
        plt.figure(figsize=(15, 10))
        
        # Sales vs Profit
        plt.subplot(2, 2, 1)
        plt.scatter(clustered_data['Sales'], clustered_data['Profit'], 
                   c=clusters, cmap='viridis')
        plt.scatter(centers[:, 0], centers[:, 3], c='red', marker='x', s=200, 
                   linewidths=3, label='المراكز')
        plt.xlabel('المبيعات')
        plt.ylabel('الربح')
        plt.title('المبيعات مقابل الربح')
        plt.legend()
        
        # Quantity vs Discount
        plt.subplot(2, 2, 2)
        plt.scatter(clustered_data['Quantity'], clustered_data['Discount'], 
                   c=clusters, cmap='viridis')
        plt.scatter(centers[:, 1], centers[:, 2], c='red', marker='x', s=200, 
                   linewidths=3, label='المراكز')
        plt.xlabel('الكمية')
        plt.ylabel('الخصم')
        plt.title('الكمية مقابل الخصم')
        plt.legend()
        
        # توزيع المجموعات
        plt.subplot(2, 2, 3)
        clustered_data['Cluster'].value_counts().plot(kind='bar')
        plt.xlabel('المجموعة')
        plt.ylabel('عدد العناصر')
        plt.title('توزيع المجموعات')
        
        plt.tight_layout()
        plt.savefig('results/kmeans_cluster_analysis.png')
        plt.close()
        
        logging.info("تم تحليل المجموعات بنجاح")
        return cluster_stats
    except Exception as e:
        logging.error(f"خطأ في تحليل المجموعات: {str(e)}")
        return None

def save_results(cluster_stats):
    """حفظ نتائج التحليل"""
    try:
        # إنشاء مجلد للنتائج إذا لم يكن موجوداً
        os.makedirs('results', exist_ok=True)
        
        # حفظ التقرير
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = f'results/kmeans_report_{timestamp}.txt'
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("تقرير تحليل المجموعات\n")
            f.write("-" * 50 + "\n\n")
            f.write("إحصائيات المجموعات:\n")
            f.write(cluster_stats.to_string())
        
        logging.info(f"تم حفظ النتائج في {report_file}")
        return True
    except Exception as e:
        logging.error(f"خطأ في حفظ النتائج: {str(e)}")
        return False

def main():
    """الدالة الرئيسية"""
    logging.info("بدء تحليل K-means")
    
    # تحميل البيانات
    df = load_data()
    if df is None:
        return False
    
    # تحضير البيانات
    data = prepare_data(df)
    if data is None:
        return False
    
    df_scaled, original_data = data
    
    # تحديد العدد الأمثل للمجموعات
    n_clusters = find_optimal_clusters(df_scaled)
    if n_clusters is None:
        return False
    
    # تنفيذ التجميع
    clustering_result = perform_clustering(df_scaled, n_clusters)
    if clustering_result is None:
        return False
    
    clusters, centers = clustering_result
    
    # تحليل المجموعات
    cluster_stats = analyze_clusters(df_scaled, clusters, centers, original_data)
    if cluster_stats is None:
        return False
    
    # حفظ النتائج
    if not save_results(cluster_stats):
        return False
    
    logging.info("اكتمل تحليل K-means بنجاح")
    return True

if __name__ == "__main__":
    main()
