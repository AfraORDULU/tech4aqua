# Tech4Aqua – Yapay zeka asistanı için proje devir notu

> **Asistana talimat:** Bu metin, bir bitirme projesinin şu ana kadarki durumunu anlatıyor. Bu sohbette projeyi devam ettirmeme yardım edeceksin. Önce metnin tamamını oku. Sonra bana (1) projeyi iki üç cümleyle özetle ve (2) "Sıradaki işler" bölümünden hangisiyle başlamak istediğimi sor. Kodu görmen gerekirse tahmin yürütme; ilgili dosyanın içeriğini benden iste. Burada yazmayan bir sonucu ya da sayıyı uydurma. Kurallar bölümündeki sızıntı ve doğrulama kurallarını hiçbir öneride bozma.

Tarih: 7 Kasım 2026. Bir sonraki toplantı: 10 Kasım 2026, 11.30 (2. toplantı: model seçimi ve yöntem belirleme).

---

## 1. Proje ne?

- **Konu:** İçme suyu numunesinde koliform bakteri uygunsuzluğunu (pozitif/negatif), laboratuvar sonucunu (24–48 saat) beklemeden tahmin etmek.
- **Asıl hedef:** Modelin eğitimde hiç görmediği su tedarik bölgelerinde (ZAPE) çalışması. Proje başlığı: "unseen water-zone için spatial domain generalization". Derin öğrenme projesi olarak tanımlandı.
- **Veri seti:** Tech4Aqua, Mendeley Data: https://data.mendeley.com/datasets/wsdv84866b/1 (Portekiz, Chaves / Montalegre / Boticas belediyeleri, 2008–2025).
- **Danışman hocanın vurguları:** Erken uyarı önemli olduğu için rutin numunelerdeki performans asıl ölçüt. Model, gelecek yılları tahmin edecek şekilde zamana göre de test edilmeli.

## 2. Depo ve çalışma düzeni

- GitHub deposu: https://github.com/AfraORDULU/tech4aqua, çalışma dalı: **`afra-dev`**. Her zaman bu dalda çalışılıyor.
- Çalışmaya başlamadan önce **Pull**, bir iş bitince hemen **Commit + Push** yapılıyor (iki kişi aynı dalda çalışıyor).
- `data/` klasörü `.gitignore` içinde, GitHub'a yüklenmiyor. Her bilgisayarda ayrıca hazırlanmalı:
  1. Mendeley paketindeki `01_Data` ve `02_Scripts` klasörleri `data/raw/` altına konur (`data/raw/01_Data/Base_Revisada_Tech4Aqua.xlsx` bulunmalı).
  2. `python -m src.build_dataset` → `data/processed/tech4aqua_labeled.csv` üretilir.
  3. Günlük hava verisi şu bağlantıdan indirilip `data/external/open_meteo_daily.csv` adıyla kaydedilir:
     `https://archive-api.open-meteo.com/v1/archive?latitude=41.74,41.82,41.69&longitude=-7.47,-7.79,-7.67&start_date=2007-12-01&end_date=2025-12-31&daily=precipitation_sum,temperature_2m_max,temperature_2m_mean&timezone=Europe/Lisbon&format=csv`
     (Konum sırası: 0 = Chaves, 1 = Montalegre, 2 = Boticas.)
- Kurulum: `python -m venv tech4aquavenv`, etkinleştir, `pip install -r requirements.txt` (torch dahil).
- Bütün komutlar proje kök klasöründen `python -m ...` şeklinde çalıştırılır.

## 3. Klasör yapısı

```
src/
  config.py            dosya yolları (DATA_DIR, DATA_PATH, RESULTS_DIR ...)
  build_dataset.py     ham kayıtlardan etiketi düzeltilmiş veri setini üretir
  data.py              load_dataset(), FEATURE_SETS (A, B, C, CHLORINE_ONLY), check_features()
  validation.py        grouped_splits() (ZAPE gruplu + arındırılmış CV), temporal_splits()
  models.py            make_model(name, seed): "LR", "RF", "LGBM", "MLP", "GROUPDRO", "CORAL"
  deep.py              TorchTabularClassifier: MLP; ERM / GroupDRO / CORAL kaybı (alan = belediye)
  weather.py           add_lag_features(): Open-Meteo'dan gecikmeli hava özellikleri
  evaluation.py        run_cv(), run_temporal(), choose_threshold(), summarize(), mean_std(), zape_bootstrap()
  data_quality_report.py, preprocessing.py, feature_groups.py   veri analizi raporu (1. toplantı)
experiments/
  e01_baseline.py              3 model × 3 özellik seti, arındırılmış ve arındırılmamış CV
  e02_routine_and_chlorine.py  rutin numune analizi + yalnız klor kıyası
  e03_temporal.py              zamansal doğrulama (genişleyen pencere, 2020 bölmesi)
  e04_weather_lags.py          gecikmeli yağış özellikleri
  e05_model_comparison.py      LR, LGBM, MLP, GroupDRO, CORAL; 3 protokol
results/<deney adı>/           her deneyin CSV çıktıları (summary.csv, oof.csv ...)
docs/2_toplanti_model_secimi.pdf   2. toplantı dokümanı
archive/                        ara rapora kadar kullanılan eski betikler (yalnızca kayıt; kullanılmıyor)
```

## 4. Veri hakkında bilinmesi gerekenler

- Modelleme verisi: koliform sonucu olan **522 numune, 82 pozitif (%15,7), 72 ZAPE**.
- Önemli sütunlar (orijinal Portekizce adlar):
  - Hedef: `Bactérias coliformes` > 0 → `y = 1`
  - ZAPE: `ZAPE NOME`; tarih: `DATA AMOSTRAGEM`; serbest klor: `Desinfetante residual` (mg/L)
  - `META_` ile başlayan sütunlar (`META_MUNICIPALITY`, `META_IS_INVESTIGATION` vb.) **asla modele girmez**; yalnızca analiz içindir.
- `load_dataset()` şu analiz sütunlarını ekler (modele girmez): `y, zape, date, municipality, year, is_investigation, chlorine_missing`.
- **Etiket düzeltmesi:** Yazarların betiği "<0.1" (tespit edilmedi) sonucunu 0.1'e çevirip yanlışlıkla pozitif sayıyordu. `build_dataset.py` "<" işaretli sonuçları 0 kabul eder ve aynı ZAPE + aynı günün kayıtlarını birleştirir.
- **Örnekleme yapısı:** 390 rutin numune (25 pozitif, %6,4) ve 132 soruşturma numunesi (57 pozitif, %43). Pozitiflerin %69,5'i soruşturma numunesi; bu numuneler bir sorun çıktıktan sonra alınıyor.
- **Belediyeler çok dengesiz:** Chaves 354 numune / 67 pozitif / 69 ZAPE (soruşturma numunelerinin hepsi burada); Montalegre 102 / 15 / 2 ZAPE; Boticas 66 / 0 / 1 ZAPE.
- 2008–2014 arasında yalnızca 7 ZAPE'den numune var. 2020–2023 arasında yılda yalnızca 1–2 pozitif var. 2025'te 38 numunenin 28'i pozitif (hepsi soruşturma).
- Klor ölçümü eksik 15 numunenin **hepsi** soruşturma numunesi. Eksik kloru "riskli" saymak sızıntıdır; eksik klor medyanla doldurulur.
- Veri setindeki hava durumu yalnızca numune günü için var. `PRECIP_LAG1_REGIONAL` gibi "LAG" sütunları bir önceki **günü** değil, aynı bölgedeki bir önceki **numunenin** gününü gösterir. Gerçek gecikmeli yağış için Open-Meteo kullanılıyor (veri setiyle uyumu: sıcaklıkta r = 0,96, yağışta r = 0,87).

## 5. Özellik setleri (`src/data.py`)

- **A:** `Desinfetante residual`, `TEMP MÁX (ºC)`, `PRECIPITAÇÃO (mm)`, `UMIDADE MÉDIA (%)`, `MES`
- **B:** `TEMP MÁX (ºC)`, `PRECIPITAÇÃO (mm)`, `UMIDADE MÉDIA (%)`, `MES`, `SIN_MES`, `COS_MES`, `FLAG_VERAO`
- **C:** A + `FLAG_CLORO_CRITICO`, `FLAG_CALOR_EXTREMO`, `FLAG_RUNOFF_RISCO`, `DEGRADACAO_TERMICA_CLORO`, `SIN_MES`, `COS_MES`, `FLAG_VERAO`
- **CHLORINE_ONLY:** yalnızca `Desinfetante residual`
- Gecikmeli hava (`src/weather.py`, numune günü dahil değil, yalnızca D-1 … D-k): `RAIN_3D`, `RAIN_7D`, `RAIN_14D`, `WET_DAYS_7D`, `DAYS_SINCE_RAIN`, `TMEAN_7D`

## 6. Doğrulama kuralları (bozulmamalı)

1. **ZAPE gruplu, arındırılmış 5 katlı CV** (`StratifiedGroupKFold`): Aynı ZAPE hem eğitimde hem testte olamaz. Ayrıca test numunesiyle aynı belediye-günü paylaşan numuneler o katta eğitimden çıkarılır (aynı gün alınan numunelerin hava verisi aynıdır). 5 tohum.
2. **Zamansal (genişleyen pencere):** Her test yılı, kendinden önceki tüm yıllarla eğitilir. `unseen_zape=True` ise test yılının ZAPE'leri eğitimden çıkarılır. Eğitimde en az 5 ZAPE ve 5 pozitif yoksa o yıl atlanır. 3 tohum.
3. **Belediye dışarıda bırakma (LOMO):** İki belediyeyle eğit, üçüncüsünde test et. Boticas'ta pozitif olmadığı için test edilemez.
- **Eşik seçimi** yalnızca eğitim verisinde, 4 katlı iç ZAPE gruplu CV ile MCC'yi en büyüten eşik olarak yapılır. Test verisi eşikte hiç kullanılmaz.
- Eksik değer doldurma ve ölçekleme Pipeline içinde, yalnızca eğitim verisinden öğrenilir.
- **Hiperparametre araması yapılmıyor** (82 pozitifle aşırı iyimser olur). Yapılacaksa iç CV içinde yapılmalı.
- `validation.py` bu kuralları `assert` ile kontrol eder; ortak ZAPE veya ortak belediye-gün bulunursa hata verir.
- Her sonuç **tüm numunelerde, rutin numunelerde ve 2025 hariç** ayrı raporlanır ve her zaman **yalnız klor sıralamasıyla** (eğitimsiz, düşük klor = yüksek risk) karşılaştırılır.
- Model karşılaştırmaları ZAPE düzeyinde eşleştirilmiş bootstrap (2000 tekrar, %95 güven aralığı) ile yapılır.

## 7. Model ayarları (`src/models.py`, `src/deep.py`)

- **LR:** median imputer + StandardScaler + LogisticRegression(class_weight="balanced", solver="liblinear", max_iter=2000)
- **RF:** median imputer + RandomForest(n_estimators=200, class_weight="balanced")
- **LGBM:** median imputer + LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=7, max_depth=4, min_child_samples=15, reg_lambda=1.0, class_weight="balanced")
- **MLP / GroupDRO / CORAL:** PyTorch; gizli katmanlar (32, 16), ReLU, dropout 0,2, Adam (lr 1e-3, weight_decay 1e-3), 300 epoch, tam batch, pos_weight ile sınıf dengeleme. GroupDRO grupları = belediye × sınıf (eta 0,1). CORAL cezası belediyeler arasında gizli katman kovaryansını hizalar (ağırlık 1,0).

## 8. Bulgular (şu ana kadar)

**Ara rapor yeniden üretildi** (ZAPE gruplu, arındırılmış CV, A + LR): ROC-AUC 0,756, PR-AUC 0,348. MCC sabit 0,5 eşiğiyle 0,29; eşik eğitimde seçilince 0,21 (eşik katlar arasında ~0,11 standart sapmayla oynuyor).

**Soru 1 – Rutin numuneler:** A + LR rutin numunelerde ROC-AUC 0,78, PR-AUC 0,20, kesinlik ~%10. Yalnız rutinle eğitilince ROC-AUC 0,83, PR-AUC 0,26. Hava + ay bilgisi soruşturma numunesi olup olmadığını kısmen tahmin ediyor (ROC-AUC 0,64), klor etmiyor (0,39).

**Soru 2 – Yalnız klor:** Rutin numunelerde ROC-AUC 0,85, PR-AUC 0,27; modelden kötü değil (fark anlamsız, klor lehine). Tüm veride model PR-AUC'de sınırda üstün (+0,066; %95 GA +0,004 ile +0,184); bu üstünlük soruşturma numunelerinden geliyor. Yasal kural (klor ≤ 0,2 mg/L) MCC 0,26–0,28.

**Soru 3 – Zaman:** Birleşik zamansal sonuç (ROC-AUC 0,74) neredeyse tamamen 2025'ten geliyor (o yıl tek başına 0,96). 2025 hariç model ROC-AUC 0,59–0,62; aynı yıllarda yalnız klor 0,70–0,78. Ağaç modelleri zamansal testte rastgeleye yakın (RF 0,51). ZAPE CV'de 2025 hariç: LR 0,67 / 0,19.

**Soru 4 – Gecikmeli yağış:** Pozitiflerden önceki 7 günde yağış daha az (medyan 2,2 mm; negatiflerde 9,8 mm), hava daha sıcak (16,7 °C; 11,7 °C). Son 3/7 gün yağışını eklemek A + LR'yi anlamlı biçimde kötüleştiriyor (ROC-AUC −0,026; %95 GA −0,048 ile −0,006). Yalnız gecikmeli hava ile model zamansal testte rastgele düzeyde. Yorum: Klorlanmış dağıtım şebekesinde yağış kaynaklı akıştan çok, sıcakta dezenfeksiyonun zayıflaması etkili görünüyor.

**Model karşılaştırması (e05, A seti, ROC-AUC):**

| Model | ZAPE CV, tümü | ZAPE CV, rutin | Gelecek yıllar, 2025 hariç, yeni ZAPE | Görülmemiş belediye (Montalegre) |
| --- | --- | --- | --- | --- |
| LR | 0,756 | 0,782 | 0,623 | 0,746 |
| LightGBM | 0,736 | 0,737 | 0,633 | 0,698 |
| MLP | 0,736 | 0,762 | 0,595 | 0,723 |
| GroupDRO | 0,732 | 0,708 | 0,659 | 0,653 |
| CORAL | 0,735 | 0,762 | 0,591 | 0,723 |
| Yalnız klor | 0,720 | 0,848 | 0,730 | 0,859 |

Hiçbir model LR'yi anlamlı biçimde geçmiyor. Rutin numunelerde GroupDRO, LR'den anlamlı biçimde kötü (−0,062). CORAL cezası modeli MLP'den neredeyse farklılaştırmadı (belediyeler çok dengesiz). Chaves test edildiğinde GroupDRO 0,79 ile öne çıktı (LR 0,78); bu tek bir bölmenin sonucu.

**MCC neden düşük:** (1) pozitifler nadir (rutinde %6,4), (2) bilgi neredeyse tamamen klorda, (3) eşik seçimi kararsız.

## 9. Toplantıda önerilecek kararlar

1. Ana model: A seti + Lojistik Regresyon. LGBM, MLP, GroupDRO, CORAL karşılaştırma modeli olarak raporda kalır.
2. Her sonuç yalnız klorla birlikte, rutin numunelerde ayrıca ve üç protokolle raporlanır.
3. Ana metrik PR-AUC. Yanında ROC-AUC, MCC (0,5 eşiği ve eğitimde seçilen eşik), kesinlik/duyarlılık, Brier skoru.
4. Gecikmeli yağış ana modele eklenmez; olumsuz sonuç olarak raporlanır.

Hocaya sorulacaklar: Derin öğrenme hedefi için "denendi, basit modeli geçmedi" bulgusu yeterli mi? Şebeke / arıtma tesisi verisi bulunabilir mi? Hedef "hangi ZAPE öncelikle örneklenmeli?" olarak değiştirilebilir mi? Kabul edilebilir alarm oranı nedir?

## 10. Sıradaki işler (öncelik sırasıyla)

1. Eşiği sabit bir alarm oranına (ör. numunelerin %10'u) veya hedef duyarlılığa göre belirleyip MCC'nin nasıl değiştiğini göstermek.
2. Klor × sıcaklık etkileşimini (sıcakta klorun hızlı bozunması) yalnız rutin numunelerle test etmek; klorun önüne geçebilecek tek aday.
3. Görevi "ZAPE-ay önceliklendirmesi" olarak kurup ilk k sıradaki isabeti ölçmek.
4. Kalibrasyon grafiği ve eşik–kesinlik–duyarlılık eğrileri.
5. Ham veri setinde kullanılmayan fiziko-kimyasal ölçüm olup olmadığını kontrol etmek.

Yeni bir deney eklenirken: `experiments/e06_<ad>.py` şeklinde yeni dosya açılır, `src/` içindeki mevcut fonksiyonlar (`load_dataset`, `run_cv`, `run_temporal`, `summarize`, `mean_std`, `zape_bootstrap`) kullanılır, çıktılar `results/e06_<ad>/` klasörüne yazılır ve README'deki deney tablosuna bir satır eklenir.
