"""
معالجة وتحضير البيانات الخام
"""
import pandas as pd
import numpy as np
from datetime import datetime
import os

def load_and_clean_data(file_path):
    """
    تحميل وتنظيف البيانات
    """
    print(f"جاري تحميل البيانات من: {file_path}")
    df = pd.read_csv(file_path, encoding='ISO-8859-1')
    
    print("\nحجم البيانات الأصلي:", df.shape)
    print("\nالأعمدة الموجودة:", list(df.columns))
    
    # حذف الصفوف التي تحتوي على قيم مفقودة
    df = df.dropna()
    print("\nحجم البيانات بعد حذف القيم المفقودة:", df.shape)
    
    # حذف الطلبات الملغاة (التي تحتوي على 'C' في رقم الفاتورة)
    df = df[~df['InvoiceNo'].astype(str).str.contains('C')]
    print("حجم البيانات بعد حذف الطلبات الملغاة:", df.shape)
    
    # حذف السجلات ذات الكمية أو السعر السالب/الصفري
    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
    print("حجم البيانات بعد تنظيف الكمية والسعر:", df.shape)
    
    return df

def add_features(df):
    """
    إضافة خصائص جديدة للبيانات
    """
    # تحويل عمود التاريخ إلى نوع datetime
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    
    # إضافة أعمدة جديدة
    df['TotalAmount'] = df['Quantity'] * df['UnitPrice']
    df['Year'] = df['InvoiceDate'].dt.year
    df['Month'] = df['InvoiceDate'].dt.month
    df['DayOfWeek'] = df['InvoiceDate'].dt.dayofweek
    df['Hour'] = df['InvoiceDate'].dt.hour
    
    return df

def main():
    """
    الدالة الرئيسية لتحضير البيانات
    """
    # التأكد من وجود المجلدات اللازمة
    data_dir = "D:/python/data_mining_project/data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    # مسار ملف البيانات الخام
    raw_data_path = os.path.join(data_dir, "train.csv")
    processed_data_path = os.path.join(data_dir, "processed_data.csv")
    
    if not os.path.exists(raw_data_path):
        print(f"خطأ: ملف البيانات الخام غير موجود في المسار: {raw_data_path}")
        return
    
    try:
        # تحميل وتنظيف البيانات
        print("=== بدء معالجة البيانات ===")
        df = load_and_clean_data(raw_data_path)
        
        # إضافة خصائص جديدة
        print("\n=== إضافة خصائص جديدة ===")
        df = add_features(df)
        
        # حفظ البيانات المعالجة
        print("\n=== حفظ البيانات المعالجة ===")
        df.to_csv(processed_data_path, index=False)
        print(f"تم حفظ البيانات المعالجة في: {processed_data_path}")
        
        # عرض معلومات إحصائية عن البيانات
        print("\n=== معلومات إحصائية عن البيانات ===")
        print("\nإحصائيات عددية:")
        print(df.describe())
        
        print("\nمعلومات عن الأعمدة:")
        print(df.info())
        
    except Exception as e:
        print("\nحدث خطأ أثناء معالجة البيانات:")
        print(f"نوع الخطأ: {type(e).__name__}")
        print(f"رسالة الخطأ: {str(e)}")

if __name__ == "__main__":
    main()
