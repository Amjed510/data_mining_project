"""
تحسين معلمات النماذج المختلفة
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

def optimize_decision_tree(X, y):
    """تحسين معلمات شجرة القرار"""
    best_params = {
        'criterion': 'gini',
        'max_depth': None,
        'min_samples_split': 2,
        'min_samples_leaf': 1
    }
    best_score = 0
    
    for criterion in ['gini', 'entropy']:
        for max_depth in [3, 5, 7, None]:
            for min_samples_leaf in [1, 2, 4]:
                clf = DecisionTreeClassifier(
                    criterion=criterion,
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    random_state=42
                )
                score = cross_val_score(clf, X, y, cv=5).mean()
                
                if score > best_score:
                    best_score = score
                    best_params = {
                        'criterion': criterion,
                        'max_depth': max_depth,
                        'min_samples_leaf': min_samples_leaf
                    }
    
    return best_params, best_score

def optimize_kmeans(X, max_k=10):
    """تحسين عدد المجموعات في K-means"""
    silhouette_scores = []
    K = range(2, max_k + 1)
    
    for k in K:
        print(f"اختبار k = {k}...")
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        score = silhouette_score(X, kmeans.labels_)
        silhouette_scores.append(score)
    
    best_k = K[np.argmax(silhouette_scores)]
    best_score = max(silhouette_scores)
    
    return best_k, best_score

def main():
    """الدالة الرئيسية لتحسين النماذج"""
    print("=== بدء عملية تحسين النماذج ===\n")
    
    # تحميل وتحضير البيانات
    print("=== تحميل وتحضير البيانات ===")
    data_path = "D:/python/data_mining_project/data/processed_data.csv"
    print(f"مسار ملف البيانات: {data_path}")
    
    try:
        import os
        print(f"هل الملف موجود؟ {os.path.exists(data_path)}")
        print("جاري قراءة البيانات...")
        df = pd.read_csv(data_path)
        print(f"تم تحميل البيانات بنجاح. الأبعاد: {df.shape}")
        print(f"الأعمدة الموجودة: {list(df.columns)}\n")
        
        # تحضير البيانات للتحليل
        print("جاري تحضير بيانات العملاء...")
        customer_features = df.groupby('CustomerID').agg({
            'Quantity': ['sum', 'mean', 'std'],
            'UnitPrice': ['mean', 'std'],
            'TotalAmount': ['sum', 'mean']
        }).round(2)
        
        # تسطيح الأعمدة
        customer_features.columns = [
            f"{col[0]}_{col[1]}" for col in customer_features.columns
        ]
        
        # تطبيع البيانات
        customer_features = (customer_features - customer_features.mean()) / customer_features.std()
        
        # تصنيف العملاء
        total_amount = customer_features['TotalAmount_sum']
        labels = pd.qcut(total_amount, q=3, labels=['Low', 'Medium', 'High'])
        
        print(f"تم تحضير بيانات {len(customer_features)} عميل")
        print(f"أعمدة البيانات المحضرة: {list(customer_features.columns)}")
        print(f"توزيع الفئات: {labels.value_counts().to_dict()}\n")
        
        print("إحصائيات البيانات:")
        print(customer_features.describe())
        print("\n")
        
        # تحسين شجرة القرار
        print("=== تحسين شجرة القرار ===")
        print("جاري تدريب النموذج...\n")
        best_dt_params, dt_score = optimize_decision_tree(customer_features, labels)
        
        print("أفضل المعلمات:")
        for param, value in best_dt_params.items():
            print(f"- {param}: {value}")
        print(f"أفضل دقة: {dt_score:.4f}\n")
        
        # تحسين Naive Bayes
        print("=== تحسين Naive Bayes ===")
        nb = GaussianNB()
        nb_scores = cross_val_score(nb, customer_features, labels, cv=5)
        print(f"متوسط الدقة: {nb_scores.mean():.4f}")
        print(f"الانحراف المعياري: {nb_scores.std():.4f}\n")
        
        # تحسين K-means
        print("=== تحسين K-means ===")
        best_k, best_silhouette = optimize_kmeans(customer_features)
        print(f"\nالعدد الأمثل للمجموعات: {best_k}")
        print(f"أفضل درجة Silhouette: {best_silhouette:.4f}\n")
        
    except Exception as e:
        print("\nحدث خطأ غير متوقع:")
        print(f"نوع الخطأ: {type(e).__name__}")
        print(f"رسالة الخطأ: {str(e)}\n")
        print("تتبع الخطأ:")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    main()
