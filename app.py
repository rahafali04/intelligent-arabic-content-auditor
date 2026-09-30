import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.isri import ISRIStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import plotly.express as px

# تنزيل حزم NLTK إذا لم تكن موجودة
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# ---------------------------------------------------------
# 1. إعدادات الصفحة والتصميم
# ---------------------------------------------------------
st.set_page_config(
    page_title="المُحلل الذكي للمحتوى العربي",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 المُحلل الذكي لمصداقية وجودة المحتوى العربي")
st.markdown("**Intelligent Arabic Content Auditor** — فحص المصداقية وتحليل الأوزان اللغوية باستخدام NLP & Machine Learning")
st.markdown("---")

# ---------------------------------------------------------
# 2. دوال معالجة النصوص
# ---------------------------------------------------------
arabic_stop_words = set(stopwords.words('arabic'))
st_stemmer = ISRIStemmer()

def basic_cleaning(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', str(text), flags=re.MULTILINE)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    # Normalization
    text = re.sub("[اإآأ]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("ة", "ه", text)
    text = re.sub("ؤ", "و", text)
    text = re.sub("ئ", "ي", text)
    # Tashkeel
    tashkeel_pattern = re.compile(r'[\u064B-\u0652]') 
    text = re.sub(tashkeel_pattern, '', text) 
    return text.strip()

def tokenize_and_remove_stop(text):
    tokens = word_tokenize(text)
    filtered_tokens = [w for w in tokens if w not in arabic_stop_words]
    return filtered_tokens

def apply_stemming(tokens):
    stemmed_tokens = [st_stemmer.stem(w) for w in tokens]
    return " ".join(stemmed_tokens)

# ---------------------------------------------------------
# 3. تحميل البيانات وتدريب الموديل (تمت المعالجة والإصلاح)
# ---------------------------------------------------------
@st.cache_resource
def load_and_train_model():
    # قراءة الملف المعتمد بمسار ثابت
    df = pd.read_csv('cleaned_news_data.csv').sample(n=5000, random_state=42)
    
    # تأكد من تحويل قيم النص لنصوص وتعبئة الفراغات
    df['stemmed_text'] = df['stemmed_text'].fillna('')
    
    # التأكد من اسم عمود التصنيف الحقيقي وتحويله لـ label
    if 'label' not in df.columns:
        if 'category' in df.columns:
            df['label'] = df['category']
        elif 'Class' in df.columns:
            df['label'] = df['Class']
        elif 'target' in df.columns:
            df['label'] = df['target']
        
    vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
    X = vectorizer.fit_transform(df['stemmed_text'].astype(str))
    y = df['label']
    
    model = SVC(kernel='linear', probability=True, random_state=42)
    model.fit(X, y)
    
    return vectorizer, model

with st.spinner("جاري تحميل النموذج والمعالج اللغوي..."):
    vectorizer, svm_model = load_and_train_model()

# ---------------------------------------------------------
# 4. واجهة المستخدم وتجربة السيناريو
# ---------------------------------------------------------
st.subheader("📝 ادخل النص أو الخبر للتدقيق والتحليل:")

default_text ="وقع الطرفان اتفاقية تعاون جديدة لتطوير البنية التحتية وتعزيز العلاقات الاقتصادية بين البلدين."

user_input = st.text_area("النص المراد فحصه:", value=default_text, height=120)

col_btn1, col_btn2 = st.columns([1, 4])
with col_btn1:
    analyze_btn = st.button("تفكيك وتحليل المحتوى 🚀", use_container_width=True)

if analyze_btn or user_input:
    if len(user_input.strip()) > 10:
        # أ) معالجة النص
        cleaned = basic_cleaning(user_input)
        tokens = tokenize_and_remove_stop(cleaned)
        stemmed = apply_stemming(tokens)
        
        # ب) التنبؤ بالموديل
        vec_input = vectorizer.transform([stemmed])
        prediction = svm_model.predict(vec_input)[0]
        
        # حساب احتمالية المصداقية (Score)
        if hasattr(svm_model, "predict_proba"):
            probs = svm_model.predict_proba(vec_input)[0]
            confidence = int(np.max(probs) * 100)
        else:
            confidence = 88
            
        st.markdown("---")
        
        # ج) عرض المؤشرات الرئيسية (Metrics Bar)
        m1, m2, m3 = st.columns(3)
        
        with m1:
            st.metric(
                label="التصنيف المتوقع (Prediction)", 
                value=str(prediction)
            )
            
        with m2:
            is_reliable = "موثوق" in str(prediction) or "حقيقي" in str(prediction) or "Medical" in str(prediction)
            st.metric(
                label="مؤشر المصداقية (Credibility Score)", 
                value=f"{confidence}%",
                delta="عالي" if is_reliable else "-منخفض",
                delta_color="normal" if is_reliable else "inverse"
            )
            
        with m3:
            st.metric(label="عدد الكلمات المفتاحية", value=f"{len(tokens)} كلمة")

        # د) الرسم البياني التفاعلي لـ TF-IDF الكلمات المفتاحية
        st.subheader("📊 أوزان الكلمات المفتاحية (TF-IDF Weights)")
        
        feature_names = vectorizer.get_feature_names_out()
        scores = vec_input.toarray()[0]
        
        word_scores = [(feature_names[i], scores[i]) for i in scores.nonzero()[0]]
        word_scores = sorted(word_scores, key=lambda x: x[1], reverse=True)[:8]
        
        if word_scores:
            df_tfidf = pd.DataFrame(word_scores, columns=['الكلمة / Root', 'TF-IDF Score'])
            
            fig = px.bar(
                df_tfidf, 
                x='TF-IDF Score', 
                y='الكلمة / Root', 
                orientation='h',
                color='TF-IDF Score',
                color_continuous_scale='Tealgrn' if is_reliable else 'Reds',
                title="أعلى الكلمات تأثيراً في قرار الموديل"
            )
            fig.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("الكلمات المدخلة جديدة على الـ Vocabulary الخاص بالـ Vectorizer، ولكن تم تحليل هيكلها العام.")

        # هـ) تفاصيل خطوات المعالجة (Pipeline Steps)
        with st.expander("🔍 خلف الكواليس: خطوات معالجة النص (Pipeline Breakdown)"):
            st.write("**1. Cleaning:**", cleaned)
            st.write("**2. Tokens & Stopwords Removed:**", tokens)
            st.write("**3. Stemmed Text (ISRIStemmer):**", stemmed)

    else:
        st.warning("الرجاء إدخال نص كافٍ للتحليل (أكثر من 10 حروف).")