# Data Mining Project | مشروع تحليل وتنقيب البيانات

## About | نبذة عن المشروع

🇺🇸 **English**:
A comprehensive data mining project focused on sales data analysis using various machine learning algorithms. The project implements Decision Trees, Naive Bayes, and K-means clustering to extract meaningful patterns and predict sales performance. Achieved 94.11% accuracy in sales prediction using Decision Trees and successfully identified 4 distinct product clusters. Built with Python, utilizing libraries such as scikit-learn, pandas, and matplotlib for analysis and visualization.

🇸🇦 **العربية**:
مشروع شامل لتنقيب البيانات يركز على تحليل بيانات المبيعات باستخدام خوارزميات التعلم الآلي المختلفة. يطبق المشروع خوارزميات شجرة القرار، نظرية بايز، وتجميع K-means لاستخراج الأنماط المفيدة والتنبؤ بأداء المبيعات. حقق المشروع دقة 94.11% في التنبؤ بالمبيعات باستخدام شجرة القرار، ونجح في تحديد 4 مجموعات متميزة من المنتجات. تم بناؤه باستخدام Python، مع الاستفادة من مكتبات مثل scikit-learn وpandas وmatplotlib للتحليل والتصور.

## وصف المشروع
هذا المشروع يهدف إلى تحليل بيانات المبيعات وتطبيق خوارزميات تنقيب البيانات المختلفة لاكتشاف الأنماط وبناء نماذج تنبؤية. يتضمن المشروع تحليل البيانات، معالجتها، وتطبيق عدة خوارزميات للتصنيف والتجميع.

## النتائج الرئيسية
1. **نموذج شجرة القرار**:
   - دقة عالية: 94.11%
   - أداء متوازن في التنبؤ بكلا الفئتين
   - F1-score: 0.94

2. **نموذج Naive Bayes**:
   - دقة: 72.15%
   - أداء أفضل في التنبؤ بالمبيعات المرتفعة
   - دقة التنبؤ بالمبيعات المرتفعة: 0.87

3. **نموذج K-means**:
   - تقسيم المنتجات إلى 4 مجموعات
   - تحليل خصائص كل مجموعة

## هيكل المشروع
```
data_mining_project/
│
├── data/                    # مجلد البيانات
│   ├── raw/                # البيانات الخام
│   └── processed/          # البيانات المعالجة
│
├── notebooks/              # Jupyter notebooks للتحليل والتوثيق
│   └── Data_Mining_Project_Documentation.ipynb
│
├── scripts/                # سكربتات المعالجة والتحليل
│   ├── prepare_data.py     # معالجة وتنظيف البيانات
│   ├── save_models.py      # تدريب وحفظ النماذج
│   └── model_testing.py    # اختبار وتقييم النماذج
│
├── models/                 # النماذج المدربة
│   ├── decision_tree_model.pkl
│   ├── naive_bayes_model.pkl
│   ├── kmeans_model.pkl
│   └── scaler.pkl
│
├── results/               # نتائج التحليل
│   ├── model_evaluation.txt
│   ├── cluster_assignments.csv
│   └── visualizations/
│
└── requirements.txt       # المكتبات المطلوبة
```

## المتطلبات
المكتبات الرئيسية المطلوبة:
```
pandas
numpy
scikit-learn
matplotlib
seaborn
jupyter
```

لتثبيت كل المكتبات:
```bash
pip install -r requirements.txt
```

## الخوارزميات المستخدمة
1. **Decision Tree (شجرة القرار)**
   - للتنبؤ بمستوى المبيعات (عالي/منخفض)
   - استخدام خصائص المنتج والعميل للتنبؤ

2. **Naive Bayes**
   - تصنيف ثنائي للمبيعات
   - تحليل احتمالية المبيعات المرتفعة

3. **K-means**
   - تجميع المنتجات في 4 مجموعات
   - تحليل خصائص كل مجموعة

## خطوات التنفيذ
1. **تحضير البيانات**
   - تحميل البيانات الخام
   - معالجة القيم المفقودة
   - تحويل التواريخ
   - إنشاء متغيرات جديدة

2. **التحليل الاستكشافي**
   - تحليل توزيع المبيعات
   - دراسة العلاقات بين المتغيرات
   - تحليل الأنماط الموسمية

3. **بناء النماذج**
   - إعداد البيانات للنمذجة
   - تدريب النماذج المختلفة
   - حفظ النماذج المدربة

4. **تقييم النماذج**
   - حساب دقة النماذج
   - تحليل تقارير التصنيف
   - دراسة خصائص المجموعات

## النتائج والتوصيات
1. **نموذج شجرة القرار** هو الأفضل للتنبؤ بمستوى المبيعات
2. تم تحديد 4 مجموعات رئيسية من المنتجات
3. هناك تباين كبير في المبيعات بين المناطق والفئات

## كيفية الاستخدام
1. استنساخ المستودع:
```bash
git clone https://github.com/Amjed510/data_mining_project.git
```

2. تثبيت المتطلبات:
```bash
pip install -r requirements.txt
```

3. تشغيل معالجة البيانات:
```bash
python scripts/prepare_data.py
```

4. تدريب النماذج:
```bash
python scripts/save_models.py
```

5. اختبار النماذج:
```bash
python scripts/model_testing.py
```

## المساهمة
نرحب بالمساهمات! يرجى اتباع الخطوات التالية:
1. Fork المستودع
2. إنشاء فرع جديد للميزة
3. تقديم pull request

## الترخيص
هذا المشروع مرخص تحت MIT License.
