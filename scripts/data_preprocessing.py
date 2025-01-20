"""
معالجة وتنظيف البيانات
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

class DataPreprocessor:
    def __init__(self, data_path):
        """
        تهيئة معالج البيانات
        :param data_path: مسار ملف البيانات
        """
        self.data_path = data_path
        self.data = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.imputer = SimpleImputer(strategy='mean')
        
    def load_data(self):
        """تحميل البيانات من الملف"""
        self.data = pd.read_csv(self.data_path)
        return self.data
        
    def handle_missing_values(self, columns=None):
        """معالجة القيم المفقودة"""
        if columns is None:
            columns = self.data.select_dtypes(include=[np.number]).columns
        self.data[columns] = self.imputer.fit_transform(self.data[columns])
        return self.data
        
    def encode_categorical_variables(self, columns):
        """تحويل المتغيرات الفئوية إلى رقمية"""
        for col in columns:
            self.data[col] = self.label_encoder.fit_transform(self.data[col])
        return self.data
        
    def scale_features(self, columns):
        """تطبيع قيم المتغيرات"""
        self.data[columns] = self.scaler.fit_transform(self.data[columns])
        return self.data
        
    def remove_duplicates(self):
        """إزالة السجلات المكررة"""
        self.data.drop_duplicates(inplace=True)
        return self.data
        
    def save_processed_data(self, output_path):
        """حفظ البيانات المعالجة"""
        self.data.to_csv(output_path, index=False)
        
    def get_data_info(self):
        """الحصول على معلومات عن البيانات"""
        info = {
            'shape': self.data.shape,
            'columns': list(self.data.columns),
            'missing_values': self.data.isnull().sum(),
            'data_types': self.data.dtypes
        }
        return info
