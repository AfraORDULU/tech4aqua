from src.config import DATA_PATH as CONFIG_DATA_PATH
import os
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from src.preprocessing import load_and_preprocess_data

def train_and_save_best_model():
    # 1. Ham veri dosyasının yolu
    data_path = str(CONFIG_DATA_PATH)
    
    print("Veri yükleniyor ve sızıntılar temizleniyor...")
    X, y, groups, feature_groups = load_and_preprocess_data(data_path)
    
    # 2. En iyi ablasyon sonucunu veren özellik kümesini seçelim 
    # (Fizikokimya + Takvim veya tüm temiz özellikler - burada en stabil 29 temiz özellik ile eğitiyoruz)
    print(f"Kullanılan toplam özellik sayısı: {X.shape[1]}")
    
    # 3. Model Pipeline'ını kur (Eksik doldurma + Ölçekleme + Random Forest)
    # Hafta 3'te en iyi skoru veren Random Forest hiperparametreleri
    best_rf_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(
            n_estimators=300, 
            random_state=42, 
            class_weight="balanced", 
            n_jobs=-1
        ))
    ])
    
    print("Model eğitiliyor...")
    best_rf_pipeline.fit(X, y)
    
    # 4. Modelleri saklamak için 'models/' klasörünü oluştur
    os.makedirs("models", exist_ok=True)
    model_output_path = "models/best_rf_model.pkl"
    
    # 5. Eğitilen modeli diske kaydet
    joblib.dump(best_rf_pipeline, model_output_path)
    print(f"Başarılı! Model diske kaydedildi: {model_output_path}")

if __name__ == "__main__":
    train_and_save_best_model()