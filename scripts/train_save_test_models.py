"""
تدريب وحفظ واختبار النماذج
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import joblib
import logging
import os
from datetime import datetime

# إعداد التسجيل
logging.basicConfig(
    filename='data/model_training.log',
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

def prepare_features(df):
    """تحضير الميزات للنماذج"""
    try:
        # معالجة القيم المفقودة
        numeric_cols = ['Sales', 'Quantity', 'Discount', 'Profit']
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].median())
        
        categorical_cols = ['Category', 'Region', 'Segment']
        for col in categorical_cols:
            df[col] = df[col].fillna(df[col].mode()[0])
        
        # تحويل التواريخ
        df['Order_Date'] = pd.to_datetime(df['Order_Date'])
        df['Year'] = df['Order_Date'].dt.year
        df['Month'] = df['Order_Date'].dt.month
        df['Quarter'] = df['Order_Date'].dt.quarter
        
        # تشفير المتغيرات الفئوية
        le = LabelEncoder()
        encoded_cols = {}
        for col in categorical_cols:
            df[f'{col}_Encoded'] = le.fit_transform(df[col])
            encoded_cols[col] = le
        
        # تطبيع البيانات العددية
        scaler = StandardScaler()
        scaled_cols = {}
        for col in numeric_cols:
            df[f'{col}_Scaled'] = scaler.fit_transform(df[[col]])
            scaled_cols[col] = scaler
        
        # تصنيف المبيعات إلى فئات
        df['Sales_Category'] = pd.qcut(df['Sales'], q=3, labels=['Low', 'Medium', 'High'])
        
        logging.info("تم تحضير الميزات بنجاح")
        return df, encoded_cols, scaled_cols
    except Exception as e:
        logging.error(f"خطأ في تحضير الميزات: {str(e)}")
        return None

def train_decision_tree(df):
    """تدريب نموذج شجرة القرار"""
    try:
        features = ['Year', 'Month', 'Quarter', 'Quantity_Scaled', 'Discount_Scaled',
                   'Category_Encoded', 'Region_Encoded', 'Segment_Encoded']
        
        X = df[features]
        y = df['Sales_Category']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        dt = DecisionTreeClassifier(random_state=42, max_depth=5)
        dt.fit(X_train, y_train)
        
        accuracy = dt.score(X_test, y_test)
        logging.info(f"دقة نموذج شجرة القرار: {accuracy:.2f}")
        
        return dt, (X_test, y_test)
    except Exception as e:
        logging.error(f"خطأ في تدريب نموذج شجرة القرار: {str(e)}")
        return None

def train_naive_bayes(df):
    """تدريب نموذج Naive Bayes"""
    try:
        features = ['Year', 'Month', 'Quarter', 'Quantity_Scaled', 'Discount_Scaled',
                   'Category_Encoded', 'Region_Encoded', 'Segment_Encoded']
        
        X = df[features]
        y = df['Sales_Category']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        nb = GaussianNB()
        nb.fit(X_train, y_train)
        
        accuracy = nb.score(X_test, y_test)
        logging.info(f"دقة نموذج Naive Bayes: {accuracy:.2f}")
        
        return nb, (X_test, y_test)
    except Exception as e:
        logging.error(f"خطأ في تدريب نموذج Naive Bayes: {str(e)}")
        return None

def train_kmeans(df):
    """تدريب نموذج K-means"""
    try:
        features = ['Quantity_Scaled', 'Sales_Scaled', 'Discount_Scaled', 'Profit_Scaled']
        X = df[features]
        
        kmeans = KMeans(n_clusters=3, random_state=42)
        kmeans.fit(X)
        
        # حساب متوسط مسافة النقاط عن مراكزها
        inertia = kmeans.inertia_
        logging.info(f"متوسط مسافة النقاط عن مراكزها: {inertia:.2f}")
        
        return kmeans, X
    except Exception as e:
        logging.error(f"خطأ في تدريب نموذج K-means: {str(e)}")
        return None

def train_apriori(df):
    """تدريب نموذج Apriori"""
    try:
        # إنشاء جدول المعاملات
        transactions = df.groupby(['Order_Date'])[['Category', 'Region', 'Segment']].agg(lambda x: list(x))
        
        # تحويل البيانات إلى تنسيق ثنائي
        encoded_vals = []
        for index, row in transactions.iterrows():
            rowset = []
            for item in row:
                rowset.extend(item)
            encoded_vals.append(rowset)
        
        # إنشاء DataFrame مشفر
        ohe = pd.get_dummies(pd.DataFrame(encoded_vals))
        
        # إيجاد مجموعات العناصر المتكررة
        frequent_itemsets = apriori(ohe, min_support=0.01, use_colnames=True)
        
        # استخراج قواعد الارتباط
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.3)
        
        logging.info(f"تم استخراج {len(rules)} قاعدة ارتباط")
        
        return rules
    except Exception as e:
        logging.error(f"خطأ في تدريب نموذج Apriori: {str(e)}")
        return None

def save_models(models, encoders, scalers):
    """حفظ النماذج والمحولات"""
    try:
        # إنشاء مجلد للنماذج
        os.makedirs('models', exist_ok=True)
        
        # حفظ النماذج
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        for name, model in models.items():
            model_path = f'models/{name}_{timestamp}.joblib'
            joblib.dump(model, model_path)
            logging.info(f"تم حفظ نموذج {name} في {model_path}")
        
        # حفظ المشفرات
        encoder_path = f'models/encoders_{timestamp}.joblib'
        joblib.dump(encoders, encoder_path)
        logging.info(f"تم حفظ المشفرات في {encoder_path}")
        
        # حفظ المطبعات
        scaler_path = f'models/scalers_{timestamp}.joblib'
        joblib.dump(scalers, scaler_path)
        logging.info(f"تم حفظ المطبعات في {scaler_path}")
        
        return True
    except Exception as e:
        logging.error(f"خطأ في حفظ النماذج: {str(e)}")
        return False

def test_models(models, test_data):
    """اختبار النماذج"""
    try:
        results = {}
        
        # اختبار شجرة القرار
        if 'decision_tree' in models and 'decision_tree' in test_data:
            dt = models['decision_tree']
            X_test, y_test = test_data['decision_tree']
            dt_accuracy = dt.score(X_test, y_test)
            results['decision_tree'] = dt_accuracy
        
        # اختبار Naive Bayes
        if 'naive_bayes' in models and 'naive_bayes' in test_data:
            nb = models['naive_bayes']
            X_test, y_test = test_data['naive_bayes']
            nb_accuracy = nb.score(X_test, y_test)
            results['naive_bayes'] = nb_accuracy
        
        # اختبار K-means
        if 'kmeans' in models and 'kmeans' in test_data:
            kmeans = models['kmeans']
            X = test_data['kmeans']
            inertia = kmeans.inertia_
            results['kmeans'] = inertia
        
        # حفظ نتائج الاختبار
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = f'results/model_test_results_{timestamp}.txt'
        
        os.makedirs('results', exist_ok=True)
        with open(results_file, 'w', encoding='utf-8') as f:
            f.write("نتائج اختبار النماذج\n")
            f.write("-" * 50 + "\n\n")
            for model_name, score in results.items():
                f.write(f"{model_name}: {score:.4f}\n")
        
        logging.info(f"تم حفظ نتائج الاختبار في {results_file}")
        return results
    except Exception as e:
        logging.error(f"خطأ في اختبار النماذج: {str(e)}")
        return None

def main():
    """الدالة الرئيسية"""
    logging.info("بدء تدريب وحفظ واختبار النماذج")
    
    # تحميل البيانات
    df = load_data()
    if df is None:
        return False
    
    # تحضير الميزات
    prepared_data = prepare_features(df)
    if prepared_data is None:
        return False
    
    df, encoders, scalers = prepared_data
    
    # تدريب النماذج
    models = {}
    test_data = {}
    
    # تدريب شجرة القرار
    dt_result = train_decision_tree(df)
    if dt_result is not None:
        models['decision_tree'], test_data['decision_tree'] = dt_result
    
    # تدريب Naive Bayes
    nb_result = train_naive_bayes(df)
    if nb_result is not None:
        models['naive_bayes'], test_data['naive_bayes'] = nb_result
    
    # تدريب K-means
    kmeans_result = train_kmeans(df)
    if kmeans_result is not None:
        models['kmeans'], test_data['kmeans'] = kmeans_result
    
    # تدريب Apriori
    rules = train_apriori(df)
    if rules is not None:
        models['apriori'] = rules
    
    # حفظ النماذج
    if not save_models(models, encoders, scalers):
        return False
    
    # اختبار النماذج
    results = test_models(models, test_data)
    if results is None:
        return False
    
    logging.info("اكتمل تدريب وحفظ واختبار النماذج بنجاح")
    return True

if __name__ == "__main__":
    main()
