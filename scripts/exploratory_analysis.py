"""
استكشاف وتحليل البيانات
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest, f_classif
import os

# تعيين نمط الرسوم البيانية
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 12

# تأكد من وجود مجلد للنتائج
os.makedirs('results', exist_ok=True)

def load_and_prepare_data():
    """تحميل ومعالجة البيانات"""
    df = pd.read_csv('data/raw/sales_data.csv')
    
    # معالجة القيم المفقودة
    numeric_cols = ['Sales', 'Quantity', 'Discount', 'Profit']
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())
    
    categorical_cols = ['Category', 'Region', 'Segment']
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
    
    return df

def analyze_numeric_distributions(df):
    """تحليل توزيع القيم العددية"""
    numeric_cols = ['Sales', 'Quantity', 'Discount', 'Profit']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('توزيع المتغيرات العددية')
    
    for i, col in enumerate(numeric_cols):
        ax = axes[i//2, i%2]
        sns.histplot(data=df, x=col, ax=ax)
        ax.set_title(f'توزيع {col}')
    
    plt.tight_layout()
    plt.savefig('results/numeric_distributions.png')
    plt.close()

def analyze_categorical_counts(df):
    """تحليل التوزيعات الفئوية"""
    categorical_cols = ['Category', 'Region', 'Segment']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('توزيع المتغيرات الفئوية')
    
    for i, col in enumerate(categorical_cols):
        sns.countplot(data=df, x=col, ax=axes[i])
        axes[i].set_title(f'توزيع {col}')
        axes[i].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('results/categorical_counts.png')
    plt.close()

def analyze_sales_trends(df):
    """تحليل اتجاهات المبيعات"""
    # تحويل تاريخ الطلب إلى datetime
    df['Order_Date'] = pd.to_datetime(df['Order_Date'])
    
    # تحليل المبيعات حسب الفئة
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='Category', y='Sales')
    plt.title('توزيع المبيعات حسب الفئة')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('results/sales_by_category.png')
    plt.close()
    
    # تحليل المبيعات حسب المنطقة
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='Region', y='Sales')
    plt.title('توزيع المبيعات حسب المنطقة')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('results/sales_by_region.png')
    plt.close()
    
    # تحليل المبيعات حسب القطاع
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='Segment', y='Sales')
    plt.title('توزيع المبيعات حسب القطاع')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('results/sales_by_segment.png')
    plt.close()

def analyze_correlations(df):
    """تحليل الارتباطات بين المتغيرات العددية"""
    numeric_cols = ['Sales', 'Quantity', 'Discount', 'Profit']
    correlation_matrix = df[numeric_cols].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('مصفوفة الارتباط بين المتغيرات العددية')
    plt.tight_layout()
    plt.savefig('results/correlation_matrix.png')
    plt.close()

def select_important_features(df):
    """تحديد أهم الميزات للتحليل"""
    # تحويل المتغيرات الفئوية إلى متغيرات رقمية
    df_encoded = pd.get_dummies(df.drop('Order_Date', axis=1), columns=['Category', 'Region', 'Segment'])
    
    # تحديد المتغيرات المستقلة والتابع
    X = df_encoded.drop(['Sales'], axis=1)
    y = df_encoded['Sales']
    
    # استخدام SelectKBest لاختيار أهم الميزات
    selector = SelectKBest(score_func=f_classif, k=5)
    selector.fit(X, y)
    
    # الحصول على درجات الأهمية
    feature_scores = pd.DataFrame({
        'Feature': X.columns,
        'Score': selector.scores_
    })
    
    # ترتيب الميزات حسب الأهمية
    feature_scores = feature_scores.sort_values('Score', ascending=False)
    
    return feature_scores

def main():
    """الدالة الرئيسية لتحليل البيانات"""
    print("بدء تحليل البيانات...")
    
    # تحميل ومعالجة البيانات
    df = load_and_prepare_data()
    
    # تحليل التوزيعات
    print("تحليل التوزيعات العددية...")
    analyze_numeric_distributions(df)
    
    print("تحليل التوزيعات الفئوية...")
    analyze_categorical_counts(df)
    
    print("تحليل اتجاهات المبيعات...")
    analyze_sales_trends(df)
    
    print("تحليل الارتباطات...")
    analyze_correlations(df)
    
    print("تحديد أهم الميزات...")
    important_features = select_important_features(df)
    
    # طباعة أهم الميزات
    print("\nأهم الميزات للتحليل:")
    print("-" * 50)
    print(important_features.head(10))
    
    print("\nتم حفظ جميع الرسوم البيانية في مجلد 'results'")

if __name__ == "__main__":
    main()
