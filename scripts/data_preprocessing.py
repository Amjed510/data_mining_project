"""
معالجة وتنظيف البيانات للمشروع
"""
import pandas as pd
import numpy as np
from datetime import datetime
import os
import logging
from sklearn.preprocessing import StandardScaler

# إعداد التسجيل
logging.basicConfig(
    filename=f'data/data_preparation.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_raw_data():
    """
    تحميل البيانات الخام
    """
    try:
        raw_data_path = os.path.join('data', 'raw', 'sales_data.csv')
        df = pd.read_csv(raw_data_path)
        logging.info(f"تم تحميل {len(df)} سجل من {raw_data_path}")
        return df
    except Exception as e:
        logging.error(f"خطأ في تحميل البيانات الخام: {str(e)}")
        return None

def remove_duplicates(df):
    """
    إزالة السجلات المكررة
    """
    initial_rows = len(df)
    df = df.drop_duplicates()
    removed_rows = initial_rows - len(df)
    logging.info(f"تم إزالة {removed_rows} سجل مكرر")
    return df

def handle_missing_values(df):
    """
    معالجة القيم المفقودة
    """
    # حساب عدد القيم المفقودة قبل المعالجة
    initial_missing = df.isnull().sum()
    
    # معالجة القيم المفقودة في الأعمدة العددية
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        df[col] = df[col].fillna(df[col].median())
    
    # معالجة القيم المفقودة في الأعمدة الفئوية
    categorical_columns = df.select_dtypes(exclude=[np.number]).columns
    for col in categorical_columns:
        df[col] = df[col].fillna(df[col].mode()[0])
    
    # حساب عدد القيم المفقودة بعد المعالجة
    final_missing = df.isnull().sum()
    
    logging.info("تم معالجة القيم المفقودة:")
    for col in df.columns:
        if initial_missing[col] > 0:
            logging.info(f"{col}: تم معالجة {initial_missing[col]} قيمة مفقودة")
    
    return df

def transform_data(df):
    """
    تحويل البيانات إلى الشكل المناسب
    """
    try:
        # تحويل التواريخ
        if 'Order_Date' in df.columns:
            df['Order_Date'] = pd.to_datetime(df['Order_Date'])
            df['Year'] = df['Order_Date'].dt.year
            df['Month'] = df['Order_Date'].dt.month
            df['Quarter'] = df['Order_Date'].dt.quarter
        
        # تحويل الأعمدة الفئوية إلى متغيرات وهمية
        categorical_columns = df.select_dtypes(include=['object']).columns
        df_encoded = pd.get_dummies(df, columns=[col for col in categorical_columns if col != 'Order_Date'])
        
        # تطبيع البيانات العددية
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        scaler = StandardScaler()
        df_encoded[numeric_columns] = scaler.fit_transform(df_encoded[numeric_columns])
        
        logging.info("تم تحويل البيانات بنجاح")
        return df_encoded, scaler
    
    except Exception as e:
        logging.error(f"خطأ في تحويل البيانات: {str(e)}")
        return None, None

def prepare_data():
    """
    الدالة الرئيسية لتحضير البيانات
    """
    logging.info("بدء عملية تحضير البيانات")
    
    # تحميل البيانات
    df = load_raw_data()
    if df is None:
        return False
    
    # تنظيف البيانات
    df = remove_duplicates(df)
    df = handle_missing_values(df)
    
    # تحويل البيانات
    df_transformed, scaler = transform_data(df)
    if df_transformed is None:
        return False
    
    # حفظ البيانات المعالجة
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    processed_data_path = os.path.join('data', f'prepared_data_{timestamp}.csv')
    df_transformed.to_csv(processed_data_path, index=False)
    logging.info(f"تم حفظ البيانات المعالجة في {processed_data_path}")
    
    # حفظ المقياس (scaler)
    if scaler is not None:
        import joblib
        scaler_path = os.path.join('models', 'scaler.pkl')
        os.makedirs('models', exist_ok=True)
        joblib.dump(scaler, scaler_path)
        logging.info(f"تم حفظ المقياس في {scaler_path}")
    
    return True

if __name__ == "__main__":
    prepare_data()
