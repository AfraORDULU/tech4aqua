# Tech4Aqua – Görülmemiş Su Bölgelerinde Koliform Riski Tahmini

İçme suyu numunesinde koliform uygunsuzluğunu, laboratuvar sonucunu (24–48 saat)
beklemeden ve modelin eğitimde hiç görmediği su tedarik bölgelerinde (ZAPE)
tahmin etmeye çalışan bitirme projesi.

Veri seti: [Tech4Aqua – Mendeley Data](https://data.mendeley.com/datasets/wsdv84866b/1)

## Kurulum

```bash
python -m venv tech4aquavenv
source tech4aquavenv/bin/activate        # Windows: tech4aquavenv\Scripts\activate
pip install -r requirements.txt
```

## Veriyi yerleştirme

Veri dosyaları GitHub'a yüklenmez (`.gitignore` içinde `data/` var).
Her bilgisayarda bir kez şunu yapın:

1. Mendeley sayfasından veri paketini indirip açın.
2. `01_Data` ve `02_Scripts` klasörlerini `data/raw/` altına kopyalayın.
   Sonuçta şu dosya bulunmalı: `data/raw/01_Data/Base_Revisada_Tech4Aqua.xlsx`
3. Etiketi düzeltilmiş veri setini üretin:

```bash
python -m src.build_dataset
```

Bu komut `data/processed/tech4aqua_labeled.csv` dosyasını oluşturur.

## Deneyleri çalıştırma

Bütün komutlar proje kök klasöründen çalıştırılır.

| Komut | Ne yapar? | Çıktı |
|---|---|---|
| `python -m src.data_quality_report` | Veri analizi ve etiket denetimi | `results/data_audit/` |
| `python -m experiments.e01_baseline` | 3 model × 3 özellik seti, arındırılmış ZAPE gruplu CV | `results/e01_baseline/` |
| `python -m experiments.e02_routine_and_chlorine` | Rutin numunelerde performans ve yalnız klor kıyası (e01 gerekli) | `results/e02_routine_and_chlorine/` |

## Klasör yapısı

```
src/
  config.py           dosya yolları
  build_dataset.py    ham kayıtlardan etiketi düzeltilmiş veri setini üretir
  data.py             veriyi yükler, özellik setlerini (A, B, C) tanımlar
  validation.py       doğrulama protokolleri (ZAPE gruplu + arındırılmış CV, zamansal bölme)
  models.py           model tanımları (LR, RF, LightGBM)
  evaluation.py       deney döngüsü, eşik seçimi, metrikler, ZAPE bootstrap
  data_quality_report.py, preprocessing.py, feature_groups.py   veri analizi raporu
experiments/          her deney ayrı bir dosya (e01_, e02_, ...)
results/              deney çıktıları
archive/              ara rapora kadar kullanılan eski betikler ve sonuçlar (yalnızca kayıt için)
app/                  demo arayüzü (henüz yeni hatta bağlanmadı)
```

## Doğrulama protokolü

- **ZAPE gruplu 5 katlı CV:** Aynı ZAPE hem eğitimde hem testte bulunamaz.
- **Arındırma:** Test numunesiyle aynı belediye-günü paylaşan numuneler o katta
  eğitimden çıkarılır. Aynı gün alınan numunelerin hava verisi neredeyse aynıdır.
- **Eşik seçimi:** Karar eşiği yalnızca eğitim verisi içinde, iç gruplu CV ile seçilir.
- **Tekrar:** Her deney 5 farklı tohumla tekrarlanır.
- Bu kuralların her biri `src/validation.py` içinde kodla doğrulanır (`assert`).
