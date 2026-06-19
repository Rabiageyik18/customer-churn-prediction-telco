import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, PowerTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import StackingClassifier

# --- PAGE CONFIGURATION & STYLING ---
st.set_page_config(
    page_title="Müşteri Churn Tahmin Modeli",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Sleek CSS (Minimalist Dark Navy Theme)
st.markdown("""
    <style>
        .main {
            background-color: #0D1B2A;
            color: #FFFFFF;
            font-family: 'Inter', sans-serif;
        }
        div[data-testid="stSidebar"] {
            background-color: #1B263B;
            border-right: 2px solid #00B4D8;
        }
        .metric-card {
            background: linear-gradient(135deg, #1E3A8A 0%, #0F172A 100%);
            border: 2px solid #3B82F6;
            border-radius: 16px;
            padding: 40px;
            box-shadow: 0 15px 25px rgba(0, 0, 0, 0.4);
            text-align: center;
            max-width: 600px;
            margin: 40px auto;
        }
        h1 {
            color: #00B4D8 !important;
            font-weight: 700 !important;
            text-align: center;
            margin-bottom: 30px;
        }
        .gauge-low {
            color: #06D6A0;
            font-size: 5rem;
            font-weight: bold;
            text-shadow: 0 0 15px rgba(6, 214, 160, 0.4);
        }
        .gauge-high {
            color: #EF4444;
            font-size: 5rem;
            font-weight: bold;
            text-shadow: 0 0 15px rgba(239, 68, 68, 0.4);
        }
    </style>
""", unsafe_allow_html=True)

# --- HELPER FUNCTIONS & MODEL TRAINING ---
@st.cache_resource
def load_data_and_train_model():
    csv_path = 'WA_Fn-UseC_-Telco-Customer-Churn.csv'
    if not os.path.exists(csv_path):
        return None, None
    
    df = pd.read_csv(csv_path)
    df = df.drop(columns=['customerID'])
    df['TotalCharges'] = df['TotalCharges'].replace({' ': '0.0'}).astype(float)
    df['Churn'] = df['Churn'].replace({'Yes': 1, 'No': 0}).astype(int)
    
    # Feature Engineering
    df['AvgMonthlyCharge'] = np.where(df['tenure'] > 0, df['TotalCharges'] / df['tenure'], df['MonthlyCharges'])
    df['ChargeIncrease'] = df['MonthlyCharges'] - df['AvgMonthlyCharge']
    
    services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    df['NumServices'] = df[services].apply(lambda x: (x == 'Yes').sum(), axis=1)
    
    df['IsNewCustomer'] = (df['tenure'] <= 6).astype(int)
    df['IsLoyalCustomer'] = (df['tenure'] >= 48).astype(int)
    df['tenure_squared'] = df['tenure'] ** 2
    
    # Shadow rules feedback variables
    df['High_Error_Risk_Newbie'] = ((df['Contract'] == 'Month-to-month') & (df['tenure'] <= 12)).astype(int)
    df['High_Error_Risk_Fiber'] = ((df['Contract'] == 'Month-to-month') & (df['tenure'] <= 15) & (df['InternetService'] == 'Fiber optic')).astype(int)
    
    X = df.drop(columns=['Churn'])
    y = df['Churn']
    
    cat_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 
                'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 
                'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']
    
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'AvgMonthlyCharge', 'ChargeIncrease', 
                'NumServices', 'IsNewCustomer', 'IsLoyalCustomer', 'tenure_squared',
                'High_Error_Risk_Newbie', 'High_Error_Risk_Fiber']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', PowerTransformer(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
        ]
    )
    
    scale = (y == 0).sum() / (y == 1).sum()
    
    # Level-0 Base Trees
    xgb = XGBClassifier(n_estimators=150, max_depth=5, learning_rate=0.05, scale_pos_weight=scale, random_state=42, eval_metric='logloss', n_jobs=-1)
    lgbm = LGBMClassifier(n_estimators=150, max_depth=5, learning_rate=0.05, scale_pos_weight=scale, random_state=42, verbose=-1, n_jobs=-1)
    cat = CatBoostClassifier(iterations=150, depth=5, learning_rate=0.05, auto_class_weights='Balanced', random_state=42, verbose=0, thread_count=-1)
    
    stacking_clf = StackingClassifier(
        estimators=[('xgb', xgb), ('lgbm', lgbm), ('cat', cat)],
        final_estimator=LogisticRegression(C=0.1, class_weight='balanced', max_iter=1000),
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        n_jobs=-1
    )
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('stacking', stacking_clf)
    ])
    
    pipeline.fit(X, y)
    return pipeline

# Train or load model
with st.spinner("🚀 Model Yükleniyor..."):
    pipeline = load_data_and_train_model()

if pipeline is None:
    st.error("HATA: 'WA_Fn-UseC_-Telco-Customer-Churn.csv' veri seti bulunamadı.")
    st.stop()

# --- APP LAYOUT ---
st.title("🎯 Müşteri Terk (Churn) Tahmin Modeli")

# Sidebar - User Inputs
st.sidebar.markdown("### 👤 Müşteri Özellikleri")

gender = st.sidebar.selectbox("Cinsiyet (gender)", ["Male", "Female"])
partner = st.sidebar.selectbox("Evli mi? (Partner)", ["Yes", "No"])
dependents = st.sidebar.selectbox("Bakmakla Yükümlü Olduğu Kişi Var mı? (Dependents)", ["No", "Yes"])
phone_service = st.sidebar.selectbox("Telefon Servisi Var mı? (PhoneService)", ["Yes", "No"])

if phone_service == "Yes":
    multiple_lines = st.sidebar.selectbox("Birden Fazla Hat Var mı? (MultipleLines)", ["No", "Yes", "No phone service"])
else:
    multiple_lines = "No phone service"

internet_service = st.sidebar.selectbox("İnternet Servis Tipi (InternetService)", ["Fiber optic", "DSL", "No"])

if internet_service != "No":
    online_security = st.sidebar.selectbox("Çevrimiçi Güvenlik Paketi (OnlineSecurity)", ["No", "Yes", "No internet service"])
    online_backup = st.sidebar.selectbox("Çevrimiçi Yedekleme (OnlineBackup)", ["No", "Yes", "No internet service"])
    device_protection = st.sidebar.selectbox("Cihaz Koruma Paketi (DeviceProtection)", ["No", "Yes", "No internet service"])
    tech_support = st.sidebar.selectbox("Teknik Destek Paketi (TechSupport)", ["No", "Yes", "No internet service"])
    streaming_tv = st.sidebar.selectbox("TV Yayını (StreamingTV)", ["No", "Yes", "No internet service"])
    streaming_movies = st.sidebar.selectbox("Film Yayını (StreamingMovies)", ["No", "Yes", "No internet service"])
else:
    online_security = "No internet service"
    online_backup = "No internet service"
    device_protection = "No internet service"
    tech_support = "No internet service"
    streaming_tv = "No internet service"
    streaming_movies = "No internet service"

contract = st.sidebar.selectbox("Kontrat Tipi (Contract)", ["Month-to-month", "One year", "Two year"])
paperless_billing = st.sidebar.selectbox("Kağıtsız Fatura Tercihi (PaperlessBilling)", ["Yes", "No"])
payment_method = st.sidebar.selectbox("Ödeme Yöntemi (PaymentMethod)", [
    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
])

st.sidebar.markdown("### 📊 Kullanım Değerleri")
tenure = st.sidebar.slider("Abonelik Süresi (Ay) (tenure)", min_value=0, max_value=72, value=12)
monthly_charges = st.sidebar.slider("Aylık Ücret ($) (MonthlyCharges)", min_value=18.0, max_value=120.0, value=70.0)
total_charges = st.sidebar.number_input("Toplam Ödenen Ücret ($) (TotalCharges)", min_value=0.0, value=float(tenure * monthly_charges))

# Decision Threshold Slider (added to sidebar as user setting)
st.sidebar.markdown("### 🛡️ Model Eşiği")
threshold = st.sidebar.slider("Karar Eşiği (Threshold)", min_value=0.10, max_value=0.90, value=0.55, step=0.01)

# --- INFERENCE ---
# Create input dictionary
input_data = {
    'gender': [gender],
    'Partner': [partner],
    'Dependents': [dependents],
    'PhoneService': [phone_service],
    'MultipleLines': [multiple_lines],
    'InternetService': [internet_service],
    'OnlineSecurity': [online_security],
    'OnlineBackup': [online_backup],
    'DeviceProtection': [device_protection],
    'TechSupport': [tech_support],
    'StreamingTV': [streaming_tv],
    'StreamingMovies': [streaming_movies],
    'Contract': [contract],
    'PaperlessBilling': [paperless_billing],
    'PaymentMethod': [payment_method],
    'tenure': [tenure],
    'MonthlyCharges': [monthly_charges],
    'TotalCharges': [total_charges]
}

df_input = pd.DataFrame(input_data)

# Compute Engineered Features
df_input['AvgMonthlyCharge'] = np.where(df_input['tenure'] > 0, df_input['TotalCharges'] / df_input['tenure'], df_input['MonthlyCharges'])
df_input['ChargeIncrease'] = df_input['MonthlyCharges'] - df_input['AvgMonthlyCharge']

services_list = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
df_input['NumServices'] = df_input[services_list].apply(lambda x: (x == 'Yes').sum(), axis=1)

df_input['IsNewCustomer'] = (df_input['tenure'] <= 6).astype(int)
df_input['IsLoyalCustomer'] = (df_input['tenure'] >= 48).astype(int)
df_input['tenure_squared'] = df_input['tenure'] ** 2

# Error shadow rules feedback
df_input['High_Error_Risk_Newbie'] = ((df_input['Contract'] == 'Month-to-month') & (df_input['tenure'] <= 12)).astype(int)
df_input['High_Error_Risk_Fiber'] = ((df_input['Contract'] == 'Month-to-month') & (df_input['tenure'] <= 15) & (df_input['InternetService'] == 'Fiber optic')).astype(int)

# Predict probability
churn_prob = pipeline.predict_proba(df_input)[0, 1]

# --- DISPLAY RESULT CARD ONLY ---
st.write(" ")
st.write(" ")

st.markdown(f"""
    <div class="metric-card">
        <h2 style='color: #00B4D8; margin-bottom: 20px;'>MÜŞTERİ TERK ETME (CHURN) OLASILIĞI</h2>
        <div class="{"gauge-high" if churn_prob >= threshold else "gauge-low"}">
            {churn_prob*100:.1f}%
        </div>
        <p style='margin-top: 25px; font-size: 1.6rem; font-weight: bold;'>
            DURUM: {"🔴 CHURN RİSKİ YÜKSEK" if churn_prob >= threshold else "🟢 GÜVENLİ / KALICI"}
        </p>
        <p style='font-size: 1rem; color: #94A3B8; margin-top: 15px;'>
            Belirlenen Karar Eşiği (Threshold): <b>{threshold:.2f}</b>
        </p>
    </div>
""", unsafe_allow_html=True)
