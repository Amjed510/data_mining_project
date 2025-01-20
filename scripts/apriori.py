"""
تنفيذ خوارزمية Apriori لاكتشاف قواعد الارتباط مع تحسين استخدام الذاكرة
"""
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

class AprioriAnalysis:
    def __init__(self, data):
        self.data = data
        
    def prepare_data(self, max_items=100):
        """
        تجهيز البيانات للتحليل مع تحديد عدد العناصر
        :param max_items: الحد الأقصى لعدد العناصر المختلفة
        """
        # تحويل البيانات إلى تنسيق سلة المشتريات
        basket = (self.data.groupby(['InvoiceNo', 'StockCode'])['Quantity']
                 .sum().unstack().reset_index().fillna(0)
                 .set_index('InvoiceNo'))
        
        # اختيار أكثر العناصر تكراراً فقط
        item_counts = basket.sum().sort_values(ascending=False)
        top_items = item_counts.head(max_items).index
        basket = basket[top_items]
        
        # تحويل البيانات إلى قيم ثنائية (0 و 1)
        self.basket_encoded = (basket > 0).astype(int)
        return self.basket_encoded
        
    def find_frequent_itemsets(self, min_support=0.01):
        """
        العثور على مجموعات العناصر المتكررة
        :param min_support: الحد الأدنى للدعم
        """
        try:
            frequent_itemsets = apriori(self.basket_encoded, 
                                      min_support=min_support, 
                                      use_colnames=True,
                                      max_len=3)  # تحديد الحد الأقصى لحجم مجموعة العناصر
            return frequent_itemsets
        except MemoryError:
            print("حدث خطأ في الذاكرة. جاري المحاولة بمعلمات أقل...")
            # محاولة مع معلمات أقل
            return self.find_frequent_itemsets(min_support=min_support*2)
        
    def generate_rules(self, frequent_itemsets, min_confidence=0.5):
        """
        توليد قواعد الارتباط
        :param min_confidence: الحد الأدنى للثقة
        """
        rules = association_rules(frequent_itemsets, 
                                metric="confidence",
                                min_threshold=min_confidence)
        return rules
        
    def evaluate_rules(self, rules):
        """تقييم القواعد المكتشفة"""
        evaluation = {
            'total_rules': len(rules),
            'avg_confidence': rules['confidence'].mean(),
            'avg_lift': rules['lift'].mean(),
            'top_rules': rules.nlargest(5, 'lift')
        }
        return evaluation
