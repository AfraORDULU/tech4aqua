import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(
    page_title="Tech4Aqua - Erken Uyarı Sistemi",
    page_icon="💧",
    layout="wide"
)

st.title("💧 Tech4Aqua: Mikrobiyolojik Su Riski Erken Uyarı Sistemi")
st.markdown("""
Bu arayüz, su dağıtım şebekesinden alınan anlık fizikokimyasal ve meteorolojik göstergeleri kullanarak,
laboratuvar analizlerini beklemeden **Koliform Bakterisi (mevzuata aykırılık)** riskini tahmin eder.
""")

# Model yükleme
@st.cache_resource
def load_trained_model():
    model_path = os.path.join("models", "best_rf_model.pkl")
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

pipeline = load_trained_model()

# Sol Panel: Saha Ölçüm Girdileri
st.sidebar.header("🧪 Anlık Ölçüm ve Saha Değerleri")

st.sidebar.subheader("Su Kimyası & Depo")
desinfetante = st.sidebar.slider("Kalıntı Klor / Dezenfektan (mg/L)", min_value=0.0, max_value=2.5, value=0.35, step=0.05)
flag_ph = st.sidebar.selectbox("pH Değeri Anormal mi?", [0, 1], format_func=lambda x: "Evet (Eşik dışı)" if x == 1 else "Hayır (Normal)")
zape_volume = st.sidebar.number_input("ZAPE Depo Hacmi (m³)", min_value=1.0, value=650.0)
zape_pop = st.sidebar.number_input("Beslenen Nüfus", min_value=1, value=1500)
delta_dias = st.sidebar.number_input("Son Numuneden Beri Geçen Gün (Delta Dias)", min_value=0, value=14)

st.sidebar.subheader("Meteorolojik Göstergeler")
temp_max = st.sidebar.slider("Maksimum Hava Sıcaklığı (°C)", min_value=-5.0, max_value=45.0, value=26.0, step=0.5)
precipitacao = st.sidebar.slider("Son Yağış Miktarı (mm)", min_value=0.0, max_value=120.0, value=8.5, step=0.5)
umidade = st.sidebar.slider("Ortalama Nem (%)", min_value=10.0, max_value=100.0, value=72.0, step=1.0)

st.sidebar.subheader("Zaman Bilgisi")
mes = st.sidebar.selectbox("Bulunulan Ay", list(range(1, 13)), index=6)

# Modele gidecek girdi tablosu
input_dict = {
    'Desinfetante residual': desinfetante,
    'FLAG_PH_ANORMAL': flag_ph,
    'ZAPE VOLUME': zape_volume,
    'ZAPE POPULACAO': zape_pop,
    'DELTA_DIAS_ZAPE': delta_dias,
    'LOG_DELTA_DIAS': np.log1p(delta_dias),
    'TEMP MÁX (ºC)': temp_max,
    'PRECIPITAÇÃO (mm)': precipitacao,
    'UMIDADE MÉDIA (%)': umidade,
    'MES': mes,
    'SIN_MES': np.sin(2 * np.pi * mes / 12),
    'COS_MES': np.cos(2 * np.pi * mes / 12),
}

input_df = pd.DataFrame([input_dict])

# Ana Ekran
col_summary, col_prediction = st.columns([1, 1])

with col_summary:
    st.subheader("📋 Girilen Saha Ölçüm Özeti")
    st.dataframe(input_df.T.rename(columns={0: "Değer"}))

with col_prediction:
    st.subheader("🚨 Risk Analizi ve Erken Uyarı")
    if pipeline is not None:
        if st.button("Risk Skorunu Hesapla", type="primary"):
            try:
                proba = pipeline.predict_proba(input_df)[0][1]
                threshold = 0.35
                
                st.metric(label="Tahmini Koliform Kontaminasyon Riski", value=f"%{proba * 100:.1f}")
                
                if proba >= threshold:
                    st.error("⚠️ **YÜKSEK RİSK: Mevzuata Aykırılık İhtimali Yüksek!**")
                    st.write("- Şebekede acil klor takviyesi yapılması önerilir.")
                    st.write("- Doğrulama amacıyla laboratuvar numunesi alınmalıdır.")
                else:
                    st.success("✅ **GÜVENLİ: Su Kalitesi Parametreleri Normal.**")
                    st.write("- Mevcut dezenfeksiyon seviyesi ve ortam koşulları mevzuata uygundur.")
                
                st.progress(min(float(proba), 1.0))
            except Exception as e:
                st.warning(f"Model girdisi işlenirken bir uyumsuzluk oluştu: {e}")
    else:
        st.error("Model dosyası (`models/best_rf_model.pkl`) bulunamadı. Lütfen önce terminalden `python src/train.py` çalıştırın.")