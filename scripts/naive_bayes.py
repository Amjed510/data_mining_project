"""
تطبيق خوارزمية Naive Bayes للتصنيف الاحتمالي
"""
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
from datetime import datetime

# إعداد التسجيل
logging.basicConfig(
    filename='data/naive_bayes_analysis.log',
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
    """تحضير البيانات للتصنيف"""
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
        for col in categorical_cols:
            df[f'{col}_Encoded'] = le.fit_transform(df[col])
        
        # تحديد المتغيرات المستقلة والتابع
        features = ['Year', 'Month', 'Quarter', 'Quantity', 'Discount',
                   'Category_Encoded', 'Region_Encoded', 'Segment_Encoded']
        
        # تصنيف المبيعات إلى فئات
        df['Sales_Category'] = pd.qcut(df['Sales'], q=3, labels=['Low', 'Medium', 'High'])
        
        X = df[features]
        y = df['Sales_Category']
        
        # تطبيع البيانات
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=features)
        
        # تقسيم البيانات
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        
        logging.info("تم تحضير البيانات بنجاح")
        return X_train, X_test, y_train, y_test, features
    except Exception as e:
        logging.error(f"خطأ في تحضير البيانات: {str(e)}")
        return None

def train_model(X_train, y_train):
    """تدريب نموذج Naive Bayes"""
    try:
        # إنشاء وتدريب النموذج
        nb = GaussianNB()
        nb.fit(X_train, y_train)
        
        logging.info("تم تدريب النموذج بنجاح")
        return nb
    except Exception as e:
        logging.error(f"خطأ في تدريب النموذج: {str(e)}")
        return None

def evaluate_model(model, X_test, y_test, features):
    """تقييم نموذج Naive Bayes"""
    try:
        # التنبؤ باستخدام مجموعة الاختبار
        y_pred = model.predict(X_test)
        
        # حساب مصفوفة الارتباك
        cm = confusion_matrix(y_test, y_pred)
        
        # إنشاء تقرير التصنيف
        report = classification_report(y_test, y_pred)
        
        # رسم مصفوفة الارتباك
        plt.figure(figsize=(10,8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('مصفوفة الارتباك')
        plt.ylabel('القيم الحقيقية')
        plt.xlabel('القيم المتنبأ بها')
        plt.savefig('results/naive_bayes_confusion_matrix.png')
        plt.close()
        
        # حساب احتمالات الفئات
        class_probabilities = model.predict_proba(X_test)
        prob_df = pd.DataFrame(class_probabilities, columns=['Low', 'Medium', 'High'])
        
        # رسم توزيع الاحتمالات
        plt.figure(figsize=(10,6))
        prob_df.boxplot()
        plt.title('توزيع احتمالات التصنيف')
        plt.ylabel('الاحتمال')
        plt.savefig('results/naive_bayes_probabilities.png')
        plt.close()
        
        logging.info("تم تقييم النموذج بنجاح")
        return report, prob_df.describe()
    except Exception as e:
        logging.error(f"خطأ في تقييم النموذج: {str(e)}")
        return None

def save_results(report, probabilities_summary):
    """حفظ نتائج التحليل"""
    try:
        # إنشاء مجلد للنتائج إذا لم يكن موجوداً
        os.makedirs('results', exist_ok=True)
        
        # حفظ التقرير
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = f'results/naive_bayes_report_{timestamp}.txt'
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("تقرير نموذج Naive Bayes\n")
            f.write("-" * 50 + "\n\n")
            f.write("نتائج التصنيف:\n")
            f.write(report)
            f.write("\n\nملخص احتمالات التصنيف:\n")
            f.write(probabilities_summary.to_string())
        
        logging.info(f"تم حفظ النتائج في {report_file}")
        return True
    except Exception as e:
        logging.error(f"خطأ في حفظ النتائج: {str(e)}")
        return False

def main():
    """الدالة الرئيسية"""
    logging.info("بدء تحليل Naive Bayes")
    
    # تحميل البيانات
    df = load_data()
    if df is None:
        return False
    
    # تحضير البيانات
    data = prepare_data(df)
    if data is None:
        return False
    
    X_train, X_test, y_train, y_test, features = data
    
    # تدريب النموذج
    model = train_model(X_train, y_train)
    if model is None:
        return False
    
    # تقييم النموذج
    evaluation = evaluate_model(model, X_test, y_test, features)
    if evaluation is None:
        return False
    
    report, probabilities_summary = evaluation
    
    # حفظ النتائج
    if not save_results(report, probabilities_summary):
        return False
    
    logging.info("اكتمل تحليل Naive Bayes بنجاح")
    return True

if __name__ == "__main__":
    main()
