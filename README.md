# Data Mining Project | مشروع تحليل وتنقيب البيانات

## About | نبذة عن المشروع

🇺🇸 **English**:
A comprehensive data mining project focused on sales data analysis using various machine learning algorithms. The project implements Decision Trees, Naive Bayes, K-means clustering, and Apriori algorithm to extract meaningful patterns and predict sales performance. Achieved 31.83% accuracy in sales prediction using Decision Trees and successfully identified 4 distinct product clusters. Built with Python, utilizing libraries such as scikit-learn, pandas, and matplotlib for analysis and visualization.

🇸🇦 **العربية**:
مشروع شامل لتنقيب البيانات يركز على تحليل بيانات المبيعات باستخدام خوارزميات التعلم الآلي المختلفة. يطبق المشروع خوارزميات شجرة القرار، نظرية بايز، تجميع K-means، وخوارزمية Apriori لاستخراج الأنماط المفيدة والتنبؤ بأداء المبيعات. حقق المشروع دقة 31.83% في التنبؤ بالمبيعات باستخدام شجرة القرار، ونجح في تحديد 4 مجموعات متميزة من المنتجات. تم بناؤه باستخدام Python، مع الاستفادة من مكتبات مثل scikit-learn وpandas وmatplotlib للتحليل والتصور.

## وصف المشروع
هذا المشروع يهدف إلى تحليل بيانات المبيعات وتطبيق خوارزميات تنقيب البيانات المختلفة لاكتشاف الأنماط وبناء نماذج تنبؤية. يتضمن المشروع تحليل البيانات، معالجتها، وتطبيق عدة خوارزميات للتصنيف والتجميع.

## النتائج الرئيسية
1. **نموذج شجرة القرار**:
   - دقة: 31.83%
   - تم تدريب النموذج على بيانات المبيعات المعالجة
   - تم حفظ النموذج مع التاريخ والوقت

2. **نموذج Naive Bayes**:
   - دقة: 29.67%
   - أداء متسق في التصنيف
   - تم حفظ النموذج للاستخدام المستقبلي

3. **نموذج K-means**:
   - متوسط مسافة النقاط عن مراكزها: 8278.80
   - تم تحديد المجموعات بنجاح
   - تم حفظ النموذج مع المشفرات والمطبعات

4. **خوارزمية Apriori**:
   - تم استخراج 111 قاعدة ارتباط
   - تحليل العلاقات بين المتغيرات
   - تم حفظ النتائج في مجلد النماذج

## هيكل المشروع
```
data_mining_project/
│
├── data/                    # مجلد البيانات
│   ├── raw/                # البيانات الخام
│   └── processed/          # البيانات المعالجة
│       └── sales_data_processed_[TIMESTAMP].csv
│
├── notebooks/              # Jupyter notebooks للتحليل والتوثيق
│   ├── 01_data_preparation.ipynb
│   ├── 02_apriori_algorithm.ipynb
│   ├── 03_decision_tree.ipynb
│   ├── 04_naive_bayes.ipynb
│   ├── 05_kmeans_clustering.ipynb
│   └── 06_train_save_test_models.ipynb
│
├── models/                 # النماذج المدربة
│   ├── decision_tree_[TIMESTAMP].joblib
│   ├── naive_bayes_[TIMESTAMP].joblib
│   ├── kmeans_[TIMESTAMP].joblib
│   ├── apriori_[TIMESTAMP].joblib
│   ├── encoders_[TIMESTAMP].joblib
│   └── scalers_[TIMESTAMP].joblib
│
├── results/               # نتائج التحليل
│   └── model_test_results_[TIMESTAMP].txt
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

4. **Apriori**
   - استخراج قواعد الارتباط بين المتغيرات
   - تحليل العلاقات بين المتغيرات

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
