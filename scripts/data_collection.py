"""
جمع وتحميل البيانات للمشروع
"""
import pandas as pd
import numpy as np
from datetime import datetime
import os
import logging

# إعداد التسجيل
logging.basicConfig(
    filename=f'data/data_collection.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def download_sample_data():
    """
    تحميل بيانات المبيعات من مصدر خارجي
    في هذه الحالة سنستخدم مجموعة بيانات Superstore
    """
    try:
        # تحميل البيانات من مصدر موثوق
        url = "https://raw.githubusercontent.com/microsoft/powerbi-desktop-samples/main/sample-sales-data/Superstore_Sales_Dataset.csv"
        df = pd.read_csv(url)
        
        # التحقق من حجم البيانات
        if len(df) < 2000:
            logging.warning(f"تم تحميل {len(df)} سجل فقط، وهو أقل من الحد الأدنى المطلوب (2000)")
            return None
        
        # حفظ البيانات الخام
        raw_data_path = os.path.join('data', 'raw', 'sales_data.csv')
        df.to_csv(raw_data_path, index=False)
        logging.info(f"تم حفظ {len(df)} سجل في {raw_data_path}")
        
        return df
    
    except Exception as e:
        logging.error(f"خطأ في تحميل البيانات: {str(e)}")
        return None

def generate_sample_data(n_samples=3000):
    """
    إنشاء بيانات عينة في حالة فشل التحميل من المصدر الخارجي
    """
    try:
        # إنشاء بيانات عشوائية
        np.random.seed(42)
        
        # تحديد القيم الممكنة للأعمدة الفئوية
        categories = ['Furniture', 'Office Supplies', 'Technology']
        regions = ['North', 'South', 'East', 'West']
        segments = ['Consumer', 'Corporate', 'Home Office']
        
        # إنشاء البيانات
        data = {
            'Order_Date': pd.date_range(start='2020-01-01', periods=n_samples),
            'Category': np.random.choice(categories, n_samples),
            'Region': np.random.choice(regions, n_samples),
            'Segment': np.random.choice(segments, n_samples),
            'Sales': np.random.uniform(100, 1000, n_samples),
            'Quantity': np.random.randint(1, 10, n_samples),
            'Discount': np.random.uniform(0, 0.5, n_samples),
            'Profit': np.random.uniform(-100, 200, n_samples)
        }
        
        df = pd.DataFrame(data)
        
        # إضافة بعض القيم المفقودة عشوائياً
        mask = np.random.random(n_samples) < 0.1
        df.loc[mask, 'Sales'] = np.nan
        
        # حفظ البيانات
        raw_data_path = os.path.join('data', 'raw', 'sales_data.csv')
        df.to_csv(raw_data_path, index=False)
        logging.info(f"تم إنشاء وحفظ {n_samples} سجل في {raw_data_path}")
        
        return df
    
    except Exception as e:
        logging.error(f"خطأ في إنشاء البيانات: {str(e)}")
        return None

def main():
    """
    الدالة الرئيسية لجمع البيانات
    """
    logging.info("بدء عملية جمع البيانات")
    
    # محاولة تحميل البيانات من المصدر الخارجي
    df = download_sample_data()
    
    # إذا فشل التحميل، قم بإنشاء بيانات عينة
    if df is None:
        logging.info("فشل تحميل البيانات من المصدر الخارجي. جاري إنشاء بيانات عينة...")
        df = generate_sample_data()
    
    if df is not None:
        # عرض ملخص للبيانات
        summary = {
            'عدد السجلات': len(df),
            'الأعمدة': list(df.columns),
            'أنواع البيانات': df.dtypes.to_dict()
        }
        
        # حفظ ملخص البيانات
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        summary_path = os.path.join('data', f'data_summary_{timestamp}.txt')
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("ملخص البيانات:\n")
            for key, value in summary.items():
                f.write(f"{key}: {value}\n")
        
        logging.info(f"تم حفظ ملخص البيانات في {summary_path}")
        return True
    
    return False

if __name__ == "__main__":
    main()
