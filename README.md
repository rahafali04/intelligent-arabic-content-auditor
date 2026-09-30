# 🧠 المُحلل الذكي لمصداقية وجودة المحتوى العربي
### Intelligent Arabic Content Auditor

تطبيق تفاعلي ذكي مبني باستخدام **Python** و **Streamlit** لتدقيق وتحليل النصوص الإخبارية والمحتوى العربي، وتقييم مدى المصداقية والتصنيف الموضوعي الفوري (مثل: Medical, Economy, Sport, Technology) باستخدام خوارزميات معالجة اللغات الطبيعية (Arabic NLP) والتعلم الآلي (Machine Learning).

---

## 🌟 الميزات الرئيسية (Key Features)

- **التصنيف الفوري ومؤشر المصداقية (Real-time Classification & Credibility Score):** إدخال النص وتحليله فورياً لتحديد فئة الخبر ومستوى الموثوقية.
- **تمثيل أوزان الكلمات بيانيًا (TF-IDF Weights Visualization):** عرض مخطط بياني تفاعلي باستخدام `Plotly Express` يُظهر الكلمات والجذور الأكثر تأثيراً في قرار النموذج.
- **تتبع خط المعالجة (Pipeline Breakdown):** إمكانية استعراض خطوات تنظيف النص، التقطيع (Tokenization)، وتصفية الكلمات الزائدة، وإرجاع الكلمات للجذور بواسطة `ISRIStemmer`.
- **مؤشرات تفاعلية (Interactive Metrics):** حساب عدد الكلمات المفتاحية وتحليل أوزان المفردات المُدخلة بشكل حيوية.

---

## 🛠️ التقنيات والمكتبات المستخدمة (Tech Stack)

- **لغة البرمجة:** Python
- **واجهة المستخدم:** Streamlit
- **معالجة النصوص (NLP):** NLTK, ISRIStemmer
- **التعلم الآلي والبيانات:** Scikit-Learn (TF-IDF & Linear SVM), Pandas, NumPy
- **الرسم البياني:** Plotly Express

---

## 📁 هيكل المشروع (Project Structure)

```text
├── app.py                                   # الملف الرئيسي لتشغيل واجهة Streamlit
├── intelligent-arabic-content-auditor.ipynb # دفتر التجارب وتحليل البيانات والنمذجة (Jupyter)
├── cleaned_news_data.csv                    # ملف البيانات الأساسي المعتمد للتطبيق
├── requirements.txt                         # قائمة المكتبات والاعتمادات البرمجية
└── README.md                                # التوثيق الشامل للمشروع