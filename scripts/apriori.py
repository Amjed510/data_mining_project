"""
تطبيق خوارزمية Apriori لاستخراج قواعد الارتباط
"""
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import logging
import os
from datetime import datetime

# إعداد التسجيل
logging.basicConfig(
    filename='data/apriori_analysis.log',
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

def prepare_data_for_apriori(df):
    """تحضير البيانات لخوارزمية Apriori"""
    try:
        # إنشاء جدول المعاملات
        transactions = df.groupby(['Order_Date'])[['Category', 'Region', 'Segment']].agg(lambda x: list(x))
        
        # تحويل البيانات إلى تنسيق ثنائي (0 و 1)
        encoded_vals = []
        for index, row in transactions.iterrows():
            rowset = []
            for item in row:
                rowset.extend(item)
            encoded_vals.append(rowset)
        
        # إنشاء DataFrame مشفر
        ohe = pd.get_dummies(pd.DataFrame(encoded_vals))
        
        logging.info("تم تحضير البيانات بنجاح")
        return ohe
    except Exception as e:
        logging.error(f"خطأ في تحضير البيانات: {str(e)}")
        return None

def find_association_rules(encoded_data, min_support=0.01, min_confidence=0.3):
    """استخراج قواعد الارتباط"""
    try:
        # إيجاد مجموعات العناصر المتكررة
        frequent_itemsets = apriori(encoded_data, 
                                  min_support=min_support, 
                                  use_colnames=True)
        logging.info(f"تم إيجاد {len(frequent_itemsets)} مجموعة متكررة")
        
        # استخراج قواعد الارتباط
        rules = association_rules(frequent_itemsets, 
                                metric="confidence", 
                                min_threshold=min_confidence)
        
        # ترتيب القواعد حسب الدعم والثقة
        rules = rules.sort_values(['support', 'confidence'], ascending=[False, False])
        
        logging.info(f"تم استخراج {len(rules)} قاعدة ارتباط")
        return rules
    except Exception as e:
        logging.error(f"خطأ في استخراج قواعد الارتباط: {str(e)}")
        return None

def save_results(rules):
    """حفظ النتائج"""
    try:
        # إنشاء مجلد للنتائج إذا لم يكن موجوداً
        os.makedirs('results', exist_ok=True)
        
        # حفظ القواعد في ملف CSV
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f'results/association_rules_{timestamp}.csv'
        rules.to_csv(output_file, index=False)
        
        # حفظ ملخص النتائج
        summary_file = f'results/apriori_summary_{timestamp}.txt'
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("ملخص تحليل قواعد الارتباط\n")
            f.write("-" * 50 + "\n")
            f.write(f"عدد القواعد المستخرجة: {len(rules)}\n")
            f.write(f"\nأفضل 5 قواعد حسب الدعم:\n")
            top_support = rules.nlargest(5, 'support')
            f.write(top_support[['antecedents', 'consequents', 'support', 'confidence']].to_string())
            f.write(f"\n\nأفضل 5 قواعد حسب الثقة:\n")
            top_confidence = rules.nlargest(5, 'confidence')
            f.write(top_confidence[['antecedents', 'consequents', 'support', 'confidence']].to_string())
        
        logging.info(f"تم حفظ النتائج في {output_file} و {summary_file}")
        return True
    except Exception as e:
        logging.error(f"خطأ في حفظ النتائج: {str(e)}")
        return False

def main():
    """الدالة الرئيسية"""
    logging.info("بدء تحليل قواعد الارتباط")
    
    # تحميل البيانات
    df = load_data()
    if df is None:
        return False
    
    # تحضير البيانات
    encoded_data = prepare_data_for_apriori(df)
    if encoded_data is None:
        return False
    
    # استخراج قواعد الارتباط
    rules = find_association_rules(encoded_data)
    if rules is None:
        return False
    
    # حفظ النتائج
    if not save_results(rules):
        return False
    
    logging.info("اكتمل تحليل قواعد الارتباط بنجاح")
    return True

if __name__ == "__main__":
    main()
