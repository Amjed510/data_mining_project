"""
فحص البيانات المعالجة
"""
import pandas as pd
import numpy as np

# قراءة البيانات المعالجة
df = pd.read_csv('data/prepared_data_20250122_045940.csv')

# طباعة معلومات عن البيانات
print("\nمعلومات عن البيانات المعالجة:")
print("-" * 50)
print(f"عدد السجلات: {len(df)}")
print(f"عدد الأعمدة: {len(df.columns)}")
print("\nالأعمدة:")
print("-" * 50)
for col in df.columns:
    print(f"{col}: {df[col].dtype}")

# التحقق من القيم المفقودة
missing_values = df.isnull().sum()
if missing_values.sum() > 0:
    print("\nالقيم المفقودة:")
    print("-" * 50)
    print(missing_values[missing_values > 0])
else:
    print("\nلا توجد قيم مفقودة في البيانات المعالجة")
