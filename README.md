# مشروع تنقيب البيانات

## وصف المشروع
هذا المشروع يتضمن تطبيق خوارزميات تنقيب البيانات المختلفة على مجموعة بيانات كبيرة.

## هيكل المشروع
```
data_mining_project/
│
├── data/                    # مجلد البيانات الخام والمعالجة
│   ├── raw/                # البيانات الخام
│   └── processed/          # البيانات المعالجة
│
├── notebooks/              # Jupyter notebooks للتحليل والتوثيق
│   ├── 1_data_exploration.ipynb
│   └── 2_data_analysis.ipynb
│
├── scripts/               # الخوارزميات الرئيسية
│   ├── apriori.py        # خوارزمية Apriori
│   ├── decision_tree.py  # خوارزمية شجرة القرار
│   ├── naive_bayes.py    # خوارزمية Naive Bayes
│   └── kmeans.py         # خوارزمية K-means
│
├── results/              # نتائج التحليل والنماذج
│
└── requirements.txt      # المكتبات المطلوبة
```

## المتطلبات
1. تثبيت المكتبات المطلوبة:
```bash
pip install -r requirements.txt
```

## الخوارزميات المستخدمة
1. Apriori Algorithm (قواعد الارتباط)
2. Decision Tree (شجرة القرار)
3. Naïve Bayes
4. K-means

## خطوات التنفيذ
1. جمع البيانات
2. معالجة وتنظيف البيانات
3. استكشاف البيانات
4. اختيار الميزات
5. تطبيق الخوارزميات
6. تقييم النماذج
