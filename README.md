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

4. Günlük hava verisini indirin (gecikmeli yağış deneyi için). Aşağıdaki bağlantı,
   Chaves, Montalegre ve Boticas için 2007-12-01 – 2025-12-31 arası günlük yağış ve
   sıcaklık verisini CSV olarak indirir. Dosyayı `data/external/open_meteo_daily.csv`
   adıyla kaydedin.

   https://archive-api.open-meteo.com/v1/archive?latitude=41.74,41.82,41.69&longitude=-7.47,-7.79,-7.67&start_date=2007-12-01&end_date=2025-12-31&daily=precipitation_sum,temperature_2m_max,temperature_2m_mean&timezone=Europe/Lisbon&format=csv

## Deneyleri çalıştırma

Bütün komutlar proje kök klasöründen çalıştırılır.

| Komut | Ne yapar? | Çıktı |
|---|---|---|
| `python -m src.data_quality_report` | Veri analizi ve etiket denetimi | `results/data_audit/` |
| `python -m experiments.e01_baseline` | 3 model × 3 özellik seti, arındırılmış ZAPE gruplu CV | `results/e01_baseline/` |
| `python -m experiments.e02_routine_and_chlorine` | Rutin numunelerde performans ve yalnız klor kıyası (e01 gerekli) | `results/e02_routine_and_chlorine/` |
| `python -m experiments.e03_temporal` | Zamana göre doğrulama: geçmişle eğit, sonraki yılları test et | `results/e03_temporal/` |
| `python -m experiments.e04_weather_lags` | Son 3/7/14 günün yağışı gibi gecikmeli hava özellikleri | `results/e04_weather_lags/` |
| `python -m experiments.e05_model_comparison` | LR, LightGBM, MLP, GroupDRO, CORAL; ZAPE CV, zamansal ve belediye dışarıda bırakma protokolleri | `results/e05_model_comparison/` |

## Klasör yapısı

```
src/
  config.py           dosya yolları
  build_dataset.py    ham kayıtlardan etiketi düzeltilmiş veri setini üretir
  data.py             veriyi yükler, özellik setlerini (A, B, C) tanımlar
  validation.py       doğrulama protokolleri (ZAPE gruplu + arındırılmış CV, zamansal bölme)
  models.py           model tanımları (LR, RF, LightGBM)
  deep.py             MLP (derin öğrenme) + alan genelleme (GroupDRO, CORAL), PyTorch
  weather.py          günlük hava verisinden gecikmeli özellikler (yalnızca numune gününden önceki günler)
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
