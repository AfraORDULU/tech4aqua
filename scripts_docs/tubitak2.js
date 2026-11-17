// TÜBİTAK 2209-A araştırma önerisi taslağı (Tech4Aqua) - sürüm 2
// Yapı, danışmanın istediği şablona (arkadaş taslağı) göre düzenlenmiştir.
const fs = require("fs");
const { Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, WidthType,
  AlignmentType, BorderStyle, ShadingType, ImageRun, LevelFormat, Footer, HeadingLevel,
  TableLayoutType } = require("docx");

const FONT = "Calibri", SZ = 19, SMALL = 17, BLUE = "1F4E79", HEAD_FILL = "D9E2F3", LABEL_FILL = "EEF3FA";
const FIG = "/home/claude/figs_tb/";
const OUTFILE = process.argv[2];

function runs(text, o = {}) {
  const out = [];
  for (const part of text.split(/(\*\*[^*]+\*\*|_[^_]+_)/)) {
    if (!part) continue;
    const b = part.startsWith("**"), it = !b && part.length > 2 && part.startsWith("_") && part.endsWith("_");
    out.push(new TextRun({ text: (b || it) ? part.slice(b ? 2 : 1, b ? -2 : -1) : part, bold: b || o.bold,
      italics: it || o.italics, font: FONT, size: o.size || SZ, color: o.color }));
  }
  return out;
}
const p = (t, o = {}) => new Paragraph({ children: runs(t, o), alignment: o.align || AlignmentType.JUSTIFIED,
  spacing: { after: o.after ?? 110, line: 264 }, keepNext: o.keepNext });
const h1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun({ text: t, bold: true, font: FONT, size: 26, color: BLUE })],
  spacing: { before: 260, after: 120 }, keepNext: true,
  border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: "8EAADB", space: 2 } } });
const h2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun({ text: t, bold: true, font: FONT, size: 22, color: "2E74B5" })],
  spacing: { before: 200, after: 90 }, keepNext: true });
const bul = (t) => new Paragraph({ children: runs(t), numbering: { reference: "b", level: 0 },
  alignment: AlignmentType.JUSTIFIED, spacing: { after: 60, line: 252 } });
const cap = (t) => new Paragraph({ children: runs(t, { size: SMALL, bold: false }), alignment: AlignmentType.LEFT,
  spacing: { before: 80, after: 80 }, keepNext: true });
const figcap = (t) => new Paragraph({ children: runs(t, { size: SMALL }), alignment: AlignmentType.JUSTIFIED, spacing: { before: 40, after: 160 } });
const img = (file, w, h) => new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 80, after: 40 },
  children: [new ImageRun({ type: "png", data: fs.readFileSync(FIG + file), transformation: { width: w, height: h } })] });

const TW = 9241; // metin genişliği (DXA)
const border = { style: BorderStyle.SINGLE, size: 4, color: "B4C6E7" };
const borders = { top: border, bottom: border, left: border, right: border };
function cell(t, w, o = {}) {
  return new TableCell({ width: { size: w, type: WidthType.DXA }, borders,
    shading: o.fill ? { type: ShadingType.CLEAR, color: "auto", fill: o.fill } : undefined,
    margins: { top: 50, bottom: 50, left: 90, right: 90 },
    children: [new Paragraph({ children: runs(t, { size: SMALL, bold: o.bold }), alignment: o.align || AlignmentType.LEFT })] });
}
function table(head, rows, widths, o = {}) {
  const total = widths.reduce((a, b) => a + b, 0);
  return new Table({ width: { size: total, type: WidthType.DXA }, columnWidths: widths, layout: TableLayoutType.FIXED,
    rows: [new TableRow({ tableHeader: true, children: head.map((h, i) => cell(h, widths[i], { fill: HEAD_FILL, bold: true })) }),
      ...rows.map(r => new TableRow({ cantSplit: true, children: r.map((c, i) => cell(c, widths[i], {
        fill: (o.labelCol && i === 0) ? LABEL_FILL : undefined, bold: o.labelCol && i === 0,
        align: (o.right || []).includes(i) ? AlignmentType.CENTER : AlignmentType.LEFT })) }))] });
}
const sp = () => new Paragraph({ children: [], spacing: { after: 60 } });

// ------------------------------------------------------------------ KAYNAKLAR
// TÜBİTAK biçimi: Soyad, A., Soyad, B. Yıl. “Başlık”, Dergi, cilt (sayı), sayfa. DOI: ...
// Her künye yayıncı / DOI / PubMed / bildiri arşivinden kontrol edilmiştir (Kasım 2026).
const REFS = [
 "Avrupa Parlamentosu ve Avrupa Birliği Konseyi. 2020. “Directive (EU) 2020/2184 of the European Parliament and of the Council of 16 December 2020 on the quality of water intended for human consumption (recast)”, Official Journal of the European Union, L 435, 1-62.",
 "baosenguo. 2020. “Kaggle-MoA-2nd-Place-Solution: 1D-CNN training notebook”, GitHub deposu, https://github.com/baosenguo/Kaggle-MoA-2nd-Place-Solution (commit bda770c; erişim tarihi: 17.11.2026).",
 "Barcia, M., Sixto, A., Cerdeiras, M. P. 2024. “Prediction of microbiological non-compliances using a Boosted Regression Trees model: application on the drinking water distribution system of a whole country”, Water Supply, 24 (4), 1080-1088. DOI: 10.2166/ws.2024.057.",
 "Brier, G. W. 1950. “Verification of forecasts expressed in terms of probability”, Monthly Weather Review, 78 (1), 1-3. DOI: 10.1175/1520-0493(1950)078<0001:VOFEIT>2.0.CO;2.",
 "Chicco, D., Jurman, G. 2020. “The advantages of the Matthews correlation coefficient (MCC) over F1 score and accuracy in binary classification evaluation”, BMC Genomics, 21 (1), 6. DOI: 10.1186/s12864-019-6413-7.",
 "Cho, K., van Merriënboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H., Bengio, Y. 2014. “Learning phrase representations using RNN encoder-decoder for statistical machine translation”, Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP), 1724-1734. DOI: 10.3115/v1/D14-1179.",
 "Curriero, F. C., Patz, J. A., Rose, J. B., Lele, S. 2001. “The association between extreme precipitation and waterborne disease outbreaks in the United States, 1948-1994”, American Journal of Public Health, 91 (8), 1194-1199. DOI: 10.2105/AJPH.91.8.1194.",
 "de Castro Paula, R., Sakurada, L., Varanda, M. J., Leitão, P., Soares, J., Teixeira, C. 2026. “Tech4Aqua: Multiparametric physicochemical and meteorological dataset with domain-driven feature engineering for microbiological water quality early warning in Northern Portuguese reservoirs”, Mendeley Data, sürüm 1. DOI: 10.17632/wsdv84866b.1.",
 "Efron, B., Tibshirani, R. J. 1993. An Introduction to the Bootstrap. Chapman & Hall, New York.",
 "Fernández-Ortega, J., Barberá, J. A., Andreo, B. 2026. “New insights into machine learning prediction techniques for real-time sanitary risk assessment in karst drinking water sources affected by faecal contamination”, Water Research, 290, 125060. DOI: 10.1016/j.watres.2025.125060.",
 "Grinsztajn, L., Oyallon, E., Varoquaux, G. 2022. “Why do tree-based models still outperform deep learning on typical tabular data?”, Advances in Neural Information Processing Systems 35 (Datasets and Benchmarks Track).",
 "Gulrajani, I., Lopez-Paz, D. 2021. “In search of lost domain generalization”, International Conference on Learning Representations (ICLR 2021).",
 "Guo, C., Pleiss, G., Sun, Y., Weinberger, K. Q. 2017. “On calibration of modern neural networks”, Proceedings of the 34th International Conference on Machine Learning, PMLR 70, 1321-1330.",
 "Heasley, C., Sanchez, J. J., Tustin, J., Young, I. 2021. “Systematic review of predictive models of microbial water quality at freshwater recreational beaches”, PLOS ONE, 16 (8), e0256785. DOI: 10.1371/journal.pone.0256785.",
 "Hersbach, H., Bell, B., Berrisford, P., Hirahara, S., Horányi, A., Muñoz-Sabater, J., Nicolas, J., Peubey, C., Radu, R., Schepers, D., Simmons, A., Soci, C., Abdalla, S., Abellan, X., Balsamo, G., Bechtold, P., Biavati, G., Bidlot, J., Bonavita, M., … Thépaut, J.-N. 2020. “The ERA5 global reanalysis”, Quarterly Journal of the Royal Meteorological Society, 146 (730), 1999-2049. DOI: 10.1002/qj.3803.",
 "Hollmann, N., Müller, S., Purucker, L., Krishnakumar, A., Körfer, M., Hoo, S. B., Schirrmeister, R. T., Hutter, F. 2025. “Accurate predictions on small data with a tabular foundation model”, Nature, 637 (8045), 319-326. DOI: 10.1038/s41586-024-08328-6.",
 "Kapoor, S., Narayanan, A. 2023. “Leakage and the reproducibility crisis in machine-learning-based science”, Patterns, 4 (9), 100804. DOI: 10.1016/j.patter.2023.100804.",
 "Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., Liu, T.-Y. 2017. “LightGBM: A highly efficient gradient boosting decision tree”, Advances in Neural Information Processing Systems 30, 3146-3154.",
 "Koçyiğit, G., Sınır, R. 2026. “Machine learning applications in predicting the microbiological quality of public fountain waters: A case study of Bayburt Province”, Black Sea Journal of Engineering and Science, 9 (5), 2365-2372. DOI: 10.34248/bsengineering.1990047.",
 "Kyritsakas, G., Boxall, J., Speight, V. 2024. “A data-driven predictive model for disinfectant residual in drinking water storage tanks”, AWWA Water Science, 6 (3), e1376. DOI: 10.1002/aws2.1376.",
 "LeChevallier, M. W., Welch, N. J., Smith, D. B. 1996. “Full-scale studies of factors related to coliform regrowth in drinking water”, Applied and Environmental Microbiology, 62 (7), 2201-2211. DOI: 10.1128/aem.62.7.2201-2211.1996.",
 "Mohammed, H., Hameed, I. A., Seidu, R. 2018. “Comparative predictive modelling of the occurrence of faecal indicator bacteria in a drinking water source in Norway”, Science of the Total Environment, 628-629, 1178-1190. DOI: 10.1016/j.scitotenv.2018.02.140.",
 "Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L., Desmaison, A., Köpf, A., Yang, E., DeVito, Z., Raison, M., Tejani, A., Chilamkurthy, S., Steiner, B., Fang, L., … Chintala, S. 2019. “PyTorch: An imperative style, high-performance deep learning library”, Advances in Neural Information Processing Systems 32, 8024-8035.",
 "Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., Duchesnay, É. 2011. “Scikit-learn: Machine learning in Python”, Journal of Machine Learning Research, 12, 2825-2830.",
 "Platt, J. C. 2000. “Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods”, Smola, A. J., Bartlett, P. L., Schölkopf, B., Schuurmans, D. (Ed.), Advances in Large Margin Classifiers, MIT Press, 61-74.",
 "Ploton, P., Mortier, F., Réjou-Méchain, M., Barbier, N., Picard, N., Rossi, V., Dormann, C. F., Cornu, G., Viennois, G., Bayol, N., Lyapustin, A., Gourlet-Fleury, S., Pélissier, R. 2020. “Spatial validation reveals poor predictive performance of large-scale ecological mapping models”, Nature Communications, 11, 4540. DOI: 10.1038/s41467-020-18321-y.",
 "Roberts, D. R., Bahn, V., Ciuti, S., Boyce, M. S., Elith, J., Guillera-Arroita, G., Hauenstein, S., Lahoz-Monfort, J. J., Schröder, B., Thuiller, W., Warton, D. I., Wintle, B. A., Hartig, F., Dormann, C. F. 2017. “Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure”, Ecography, 40 (8), 913-929. DOI: 10.1111/ecog.02881.",
 "Sagawa, S., Koh, P. W., Hashimoto, T. B., Liang, P. 2020. “Distributionally robust neural networks for group shifts: On the importance of regularization for worst-case generalization”, International Conference on Learning Representations (ICLR 2020).",
 "Saito, T., Rehmsmeier, M. 2015. “The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets”, PLOS ONE, 10 (3), e0118432. DOI: 10.1371/journal.pone.0118432.",
 "Searcy, R. T., Boehm, A. B. 2021. “A day at the beach: Enabling coastal water quality prediction with high-frequency sampling and data-driven models”, Environmental Science & Technology, 55 (3), 1908-1918. DOI: 10.1021/acs.est.0c06742.",
 "Sokolova, E., Ivarsson, O., Lillieström, A., Speicher, N. K., Rydberg, H., Bondelind, M. 2022. “Data-driven models for predicting microbial water quality in the drinking water source using E. coli monitoring and hydrometeorological data”, Science of the Total Environment, 802, 149798. DOI: 10.1016/j.scitotenv.2021.149798.",
 "Stocker, M. D., Pachepsky, Y. A., Hill, R. L. 2022. “Prediction of E. coli concentrations in agricultural pond waters: Application and comparison of machine learning algorithms”, Frontiers in Artificial Intelligence, 4, 768650. DOI: 10.3389/frai.2021.768650.",
 "Sun, B., Saenko, K. 2016. “Deep CORAL: Correlation alignment for deep domain adaptation”, Computer Vision - ECCV 2016 Workshops, Lecture Notes in Computer Science 9915, Springer, 443-450. DOI: 10.1007/978-3-319-49409-8_35.",
 "Volf, G., Sušanj Čule, I., Atanasova, N., Zorko, S., Ožanić, N. 2025. “Explaining and predicting microbiological water quality for sustainable management of drinking water treatment facilities”, Sustainability, 17 (15), 6659. DOI: 10.3390/su17156659.",
 "White, K., Dickson-Anderson, S., Majury, A., McDermott, K., Hynds, P., Brown, R. S., Schuster-Wallace, C. 2021. “Exploration of E. coli contamination drivers in private drinking water wells: An application of machine learning to a large, multivariable, geo-spatio-temporal dataset”, Water Research, 197, 117089. DOI: 10.1016/j.watres.2021.117089.",
 "Willard, J. D., Varadharajan, C., Jia, X., Kumar, V. 2025. “Time series predictions in unmonitored sites: A survey of machine learning techniques in water resources”, Environmental Data Science, 4, e7. DOI: 10.1017/eds.2024.14.",
 "World Health Organization. 2022. Guidelines for Drinking-Water Quality: Fourth Edition Incorporating the First and Second Addenda. World Health Organization, Geneva.",
 "Zhou, K., Liu, Z., Qiao, Y., Xiang, T., Loy, C. C. 2023. “Domain generalization: A survey”, IEEE Transactions on Pattern Analysis and Machine Intelligence, 45 (4), 4396-4415. DOI: 10.1109/TPAMI.2022.3195549.",
 "Zhu, M., Wang, J., Yang, X., Zhang, Y., Zhang, L., Ren, H., Wu, B., Ye, L. 2022. “A review of the application of machine learning in water quality evaluation”, Eco-Environment & Health, 1 (2), 107-116. DOI: 10.1016/j.eehl.2022.06.001.",
];
fs.writeFileSync("/home/claude/refs_tubitak2.json", JSON.stringify(REFS, null, 1));

// ------------------------------------------------------------------ İÇERİK
const children = [
  new Paragraph({ children: [new TextRun({ text: "TÜBİTAK 2209-A", bold: true, font: FONT, size: 34, color: BLUE })], spacing: { after: 60 },
    border: { bottom: { style: BorderStyle.SINGLE, size: 8, color: "8EAADB", space: 4 } } }),
  h2("Genel Bilgiler"),
  table(["Alan", "Bilgi"], [
    ["Proje başlığı (Türkçe)", "Görülmemiş İçme Suyu Bölgelerinde Koliform Riskinin Erken Tahmini: Sızıntısız Değerlendirme ve Bölge Geçmişiyle Hızlı Uyum Sağlayan Derin Öğrenme Modelleri"],
    ["Proje başlığı (İngilizce)", "Early Prediction of Coliform Risk in Unseen Drinking Water Zones: Leakage-Free Evaluation and Deep Learning Models that Adapt Quickly Using Zone History"],
    ["Araştırmacılar", "Afra Ordulu; Nilüfer Öden"],
    ["Kurum / bölüm", "İzmir Bakırçay Üniversitesi / Mühendislik ve Mimarlık Fakültesi / Bilgisayar Mühendisliği"],
    ["Akademik danışman", "Doç. Dr. Muhammet Emin Şahin"],
    ["Önerilen destek dönemi", "8 ay"],
  ], [2800, 6441], { labelCol: true }),
  sp(),

  h1("ÖZET"),
  p("Bu araştırma, içme suyu dağıtım şebekesinden alınan numunelerde koliform uygunsuzluğunun, laboratuvar sonucu (24-48 saat) beklenmeden ve modelin daha önce hiç görmediği su dağıtım bölgelerinde (ZAPE) tahmin edilmesini amaçlamaktadır. Ana veri kaynağı, Kuzey Portekiz'deki üç belediyeye ait, 2008-2025 yıllarını kapsayan açık erişimli Tech4Aqua veri setidir (522 numune, 82 uygunsuz, 72 ZAPE). Model girdileri numune anında bilinen bilgilerle sınırlıdır: serbest klor, hava koşulları, takvim ve bölgenin sonucu daha önce gelmiş ölçümleri. Geliştirilecek yaklaşım laboratuvar analizinin yerini alan bir sistem değil, sınırlı numune kapasitesini riskli bölgelere yönlendirmeye yardımcı olacak bir karar destek prototipidir."),
  p("Ön çalışmalarda yayımlanmış veri setinde bir etiket hatası ve aynı gün alınan numunelerden kaynaklanan bir veri sızıntısı bulunmuş ve giderilmiştir. Sızıntısız bir değerlendirme çerçevesinde Lojistik Regresyon, LightGBM, önceden eğitilmiş tablo modeli TabPFN, MLP, 1D-CNN, alan genelleme yöntemleri (GroupDRO, CORAL) ve bölge geçmişini dizi olarak okuyan GRU modeli, eğitim gerektirmeyen klor sıralamasıyla karşılaştırılmıştır. Tüm numunelerde en yüksek başarı TabPFN'in bölge geçmişiyle kullanılmasıyla elde edilmiştir (ROC-AUC 0,88; PR-AUC 0,56); ancak rutin numunelerde hiçbir model klor sıralamasını anlamlı biçimde geçememiştir ve bölge geçmişinden gelen kazancın büyük kısmının rutin ve soruşturma numunelerinin ayrışmasından kaynaklandığı görülmüştür. TÜBİTAK kapsamında bu bulgular görülmemiş bölge, gelecek yıl ve görülmemiş belediye senaryolarında sabit protokollerle yeniden sınanacak; kalibrasyon, alarm bütçesi ve hata analizleri raporlanacak ve bir erken uyarı demo uygulaması geliştirilecektir. Beklenen çıktı, hangi bilginin hangi koşulda tahmine katkı sağladığını ve hangi sınırlılıkların sürdüğünü gösteren yeniden üretilebilir bir deney raporu ile çalışan bir prototiptir."),
  p("**Anahtar Kelimeler:** içme suyu kalitesi, koliform, erken uyarı, alan genelleme, derin öğrenme, TabPFN, veri sızıntısı"),

  h1("1. ÖZGÜN DEĞER"),
  h2("1.1. Konu ve Kapsam"),
  p("Koliform bakteriler, içme suyunda fekal kirlenmenin ve dezenfeksiyon yetersizliğinin göstergesi olarak izlenir (World Health Organization, 2022). Avrupa Birliği içme suyu direktifi koliform bakteriyi 100 mL'de 0 değeriyle izlenen göstergeler arasında saymaktadır (Avrupa Parlamentosu ve Avrupa Birliği Konseyi, 2020). Rutin izlemede numunenin sonucu laboratuvardan ancak 24-48 saat sonra gelir; bu sürede uygunsuz su tüketilmeye devam edebilir. Ayrıca su idarelerinin numune ve laboratuvar kapasitesi sınırlıdır. Numune alındığı anda bilinen bilgilerle uygunsuzluk riskinin önceden tahmin edilmesi, bu kapasitenin riskli bölgelere yönlendirilmesini ve sonuç gelmeden önlem alınmasını kolaylaştırabilir."),
  p("Önerilen proje, dağıtım şebekesinden alınan bir numunenin koliform açısından uygunsuz olup olmayacağının ikili sınıflandırma olarak tahmin edilmesini konu edinmektedir. Model girdileri yalnızca numune anında bilinen bilgilerle sınırlandırılacaktır: serbest klor kalıntısı, günün ve önceki günlerin hava koşulları, takvim ve aynı bölgenin sonucu daha önce gelmiş numuneleri. Projenin odağı, modelin daha önce hiç görmediği bir bölgede ne kadar başarılı olduğudur; çünkü yeni izlenmeye başlanan veya az numune alınan bölgeler, erken uyarının en çok gerektiği yerlerdir. Az sayıda geçmiş ölçümü olan bir bölgenin, uzun süredir izlenen bir bölgeden farklı bir problem oluşturduğu dikkate alınacaktır."),
  p("Araştırmanın temel veri kaynağı, Mendeley Data üzerinde açık erişimli yayımlanan Tech4Aqua veri setidir (de Castro Paula vd., 2026). Etiket düzeltmesinden sonra veri, Chaves, Montalegre ve Boticas belediyelerindeki 72 ZAPE'den 2008-2025 yılları arasında alınmış 522 numune içermektedir; bunların 82'si (%15,7) uygunsuzdur. Numunelerin 390'ı rutin izleme (25 uygunsuz, %6,4), 132'si bir sorun sonrasında alınan soruşturma numunesidir (57 uygunsuz). Rutin numuneler yalnızca 7 ZAPE'den gelmektedir ve 33 ZAPE'de yalnızca bir numune bulunmaktadır. Bu dağılım, sonuçların numune türüne göre ayrı raporlanması ve belirsizliklerin açıkça verilmesi gerektiğini göstermektedir."),
  p("Proje çıktıları laboratuvar analizinin yerine geçen bir tanı aracı olarak değil, numune önceliklendirmesine yardımcı olacak risk tahminleri olarak tanımlanacaktır. Veri Kuzey Portekiz'deki üç belediyeyle sınırlı olduğundan, başka bir ülkenin veya şebekenin verisiyle bağımsız dış test yapılmadıkça genellenebilirlik iddia edilmeyecektir. Koliform derişiminin tahmini, şebeke hidrolik modellemesi ve patojen tespiti kapsam dışıdır."),

  h2("1.2. Literatür Özeti"),
  p("**Koliform oluşumunun belirleyicileri.** LeChevallier vd. (1996), Kuzey Amerika'daki 31 su sisteminde 18 ay süren saha çalışmasında koliform oluşumunun filtrasyon, sıcaklık, dezenfektan türü ve düzeyi, asimile edilebilir organik karbon, korozyon kontrolü ve işletme koşullarıyla ilişkili olduğunu göstermiştir. Bu çalışmada su sıcaklığı 15 °C'nin üzerindeyken koliform anlamlı biçimde daha sık görülmüş, dezenfektan kalıntısını 0,5 mg/L'nin altında tutan sistemlerde belirgin biçimde daha fazla koliform bulunmuş ve tek bir etkenin tüm olayları açıklayamadığı vurgulanmıştır. Curriero vd. (2001), ABD'de 1948-1994 arasındaki su kaynaklı salgınların %51'inden önce 90. yüzdeliğin üzerinde yağış görüldüğünü bildirmiştir. Bu çalışmalar klor, sıcaklık ve yağışın neden aday değişken olduğunu açıklar; ancak tek bir numunenin sonucunu, özellikle görülmemiş bir bölgede tahmin etme gücünü sınamazlar."),
  p("**İçme suyunda makine öğrenmesiyle mikrobiyolojik kalite tahmini.** Konuya en yakın çalışmada Barcia vd. (2024), bir ülkenin tüm içme suyu dağıtım sisteminde teslim noktasındaki mikrobiyolojik uygunsuzlukları (toplam koliform, E. coli, Pseudomonas aeruginosa) güçlendirilmiş regresyon ağaçlarıyla modellemiş; AUC 0,77 elde etmiş ve en etkili değişkenlerin coğrafi konum, serbest klor ve yerleşim nüfusu olduğunu bulmuştur. Ham su kaynaklarında yapılan çalışmalarda Volf vd. (2025), Hırvatistan'daki Butoniga barajında toplam koliform ve E. coli'yi yedi gün önceden tahmin etmiş; 10 katlı çapraz doğrulamayla model ağaçlarında toplam koliform için 0,72, E. coli için 0,48 korelasyon elde etmiş ve modellerin tepe değerleri düşük tahmin ettiğini bildirmiştir. Sokolova vd. (2022), İsveç'te Göta älv nehrindeki su alma noktasında E. coli'yi bir gün önceden tahmin etmiş; çok değişkenli modellerin tek değişkenli yöntemlerden iyi olduğunu, ancak rastgele orman ve otomatik makine öğrenmesi modellerinin aşırı öğrenme eğilimi gösterdiğini raporlamıştır. Mohammed vd. (2018) Norveç'te bir içme suyu kaynağında modelin düşük derişimleri tahmin edemediğini, Fernández-Ortega vd. (2026) ise İspanya'da iki karst kaynağında 194 E. coli numunesiyle on sınıflandırıcıyı beş sanitasyon risk düzeyi için karşılaştırmıştır. Türkiye'den Koçyiğit ve Sınır (2026), Bayburt'taki beş çeşmeden 2014-2024 arasında alınmış 476 analiz raporunda %80/%20 rastgele bölmeyle bazı sınıflandırıcılarda %100 doğruluk bildirmiştir."),
  p("Daha geniş ölçekte White vd. (2021), Kanada'nın Ontario eyaletinde 253.136 özel kuyudan alınmış 795.023 E. coli ölçümünü makine öğrenmesiyle incelemiştir. Rekreasyon suları için yapılan sistematik derlemede Heasley vd. (2021), 53 çalışmanın %83'ünün ABD'de yapıldığını ve en sık kullanılan yöntemin çoklu doğrusal regresyon (%70) olduğunu göstermiştir. Searcy ve Boehm (2021), uzun geçmiş verisi olmayan plajlarda kısa süreli yüksek frekanslı örneklemeyle eğitilen modellerin kullanılabileceğini göstermiştir; Stocker vd. (2022) ise sulama göletlerinde algoritmalar arasındaki farkların çoğu durumda anlamlı olmadığını bulmuştur (genel bir derleme için bkz. Zhu vd., 2022). Bu çalışmalar, mikrobiyolojik kalitenin makine öğrenmesiyle tahmin edilebileceğini göstermekte; ancak çoğu tek bir kaynak veya az sayıda istasyonda yapılmakta ve aynı istasyonların verisiyle oluşturulan doğrulamada değerlendirilmektedir."),
  p("**Dezenfektan kalıntısı ve risk sıralaması.** Kyritsakas vd. (2024), beş milyondan fazla kişiye hizmet veren bir su idaresinde depolama tanklarını bir ay sonraki düşük dezenfektan riski açısından sıralamış; hiç kullanılmamış son yılın verisinde ilk 20 sıradaki tankların %90'ından fazlasında düşük kalıntı görülmüştür. Bu çalışma, kesin bir eşik yerine riske göre sıralamanın ve sınırlı denetim kapasitesine göre değerlendirmenin su idareleri için pratik değerini göstermektedir; önerilen projedeki alarm bütçesi analizi bu yaklaşımla uyumludur."),
  p("**Değerlendirme, veri sızıntısı ve görülmemiş bölgeler.** Mekânsal veya zamansal yapısı olan verilerde rastgele bölünmüş doğrulama iyimser sonuç verir; gruplu, blok veya zamansal doğrulama önerilmektedir (Roberts vd., 2017). Ploton vd. (2020), mekânsal yapıyı yok sayan doğrulamada modelin değişkenliğin yarısından fazlasını açıklıyor göründüğünü, mekânsal doğrulamada ise tahmin gücünün neredeyse sıfıra indiğini göstermiştir. Kapoor ve Narayanan (2023), veri sızıntısının makine öğrenmesi tabanlı bilimde yaygın bir tekrar üretilebilirlik sorunu olduğunu ortaya koymuştur. Su kaynakları alanında Willard vd. (2025), izlenmeyen noktalarda tahmin üzerine yaptıkları derlemede çalışmaların çoğunun ABD'de çok sayıda istasyona dayanan derin öğrenme çerçevelerine odaklandığını ve farklı yöntem sınıflarının ayrıntılı karşılaştırmasının eksik olduğunu belirtmiştir. Makine öğrenmesi literatüründe bu sorun alan genelleme olarak ele alınır (Zhou vd., 2023): GroupDRO en kötü durumdaki grubun hatasını küçültmeyi (Sagawa vd., 2020), CORAL alanlar arasındaki temsil dağılımlarını hizalamayı amaçlar (Sun ve Saenko, 2016). Ancak Gulrajani ve Lopez-Paz (2021), dikkatle kurulmuş sıradan eğitimin bu yöntemlerin çoğuyla benzer başarı verdiğini göstermiştir."),
  p("**Küçük tablo verisinde derin öğrenme.** Grinsztajn vd. (2022), 45 veri setinde ağaç tabanlı modellerin orta büyüklükteki (yaklaşık 10 bin örnek) tablo verisinde derin öğrenme modellerinden üstün kaldığını göstermiştir; bu projenin verisi bunun çok altındadır. Buna karşılık TabPFN, çok sayıda yapay veri setiyle önceden eğitilmiş bir transformer modelidir; yeni veri setinde ağırlıklarını güncellemeden, eğitim örneklerini bağlam olarak kullanarak tahmin yapar ve küçük veri setlerinde güçlü sonuçlar raporlamıştır (Hollmann vd., 2025). GRU gibi yinelemeli ağlar ise bir bölgenin geçmiş ölçümlerini sırasıyla okuyabilir (Cho vd., 2014). Sinir ağlarının olasılık çıktıları çoğu zaman iyi kalibre değildir (Guo vd., 2017); karar desteğinde kullanılacak bir risk yüzdesinin güvenilirliğinin ayrıca ölçülmesi gerekir."),
  p("**Mevcut durum ve araştırma gereksinimi.** İncelenen literatürde içme suyu kaynaklarında ve dağıtım sistemlerinde mikrobiyolojik kalitenin makine öğrenmesiyle tahmini üzerine önemli bir birikim bulunmaktadır. Buna karşılık (i) klorlanmış dağıtım şebekesinde numune düzeyinde uygunsuzluk tahmininin görülmemiş bölge, gelecek yıl ve görülmemiş belediye senaryolarında sızıntısız biçimde sınanması, (ii) modellerin laboratuvar beklemeden kullanılabilen en basit referans olan klor sıralamasıyla belirsizlik aralıklarıyla karşılaştırılması ve (iii) küçük veride sıfırdan eğitilen sinir ağları, dizi modelleri ve önceden eğitilmiş tablo modellerinin aynı protokolde karşılaştırılarak yeni bir bölgenin ilk ölçümleriyle ne kadar hızlı uyum sağlanabildiğinin ölçülmesi incelenen kaynaklarda sınırlıdır. Önerilen projede bu üç konu, numune türüne göre ayrılmış ölçütlerle birlikte ele alınacaktır."),

  h2("1.3. Özgün Değer"),
  p("Projenin özgün değeri, içme suyu dağıtım şebekesinde koliform uygunsuzluğu tahminini, modelin hiç görmediği bölgelere genelleme açısından sızıntısız bir değerlendirme çerçevesiyle ve eğitim gerektirmeyen klor sıralamasına karşı ölçülen kazançla birleştirmesidir. Ön çalışmalar kapsamında yayımlanmış veri setindeki bir etiket hatası ve aynı gün kaynaklı bir veri sızıntısı tespit edilerek giderilmiş; klasik, önceden eğitilmiş ve derin öğrenme modelleri üç test senaryosunda karşılaştırılmış ve bölge geçmişinin katkısı ayrıştırılmıştır. Bu birikim teknik uygulanabilirliğe temel oluşturmakta; önerilen araştırma, ön bulguların sabit protokollerle doğrulanmasını, rutin izleme koşullarında modelin gerçek katkısının belirlenmesini ve sonuçların su idaresinin karar diliyle (alarm bütçesi, risk yüzdesi) raporlanmasını hedeflemektedir."),
  p("**Araştırma sorusu:** “Numune anında bilinen klor, hava, takvim ve bölge geçmişi bilgileri, modelin hiç görmediği içme suyu bölgelerinde koliform uygunsuzluğunu eğitimsiz klor sıralamasından daha iyi tahmin etmeyi sağlar mı; sağlıyorsa bu kazanç hangi model türüyle ve hangi koşullarda (numune türü, bölgeden bilinen ölçüm sayısı) ortaya çıkar?”"),
  bul("**Hipotez 1:** Yalnızca klor, hava ve takvim bilgisiyle eğitilen modeller, görülmemiş bölgelerde ve gelecek yıllarda klor sıralamasını anlamlı biçimde geçemez (ön çalışma bunu desteklemektedir)."),
  bul("**Hipotez 2:** Bölge geçmişi, özellikle bölgeden az sayıda (1-5) sonuç bilindiğinde tahmini iyileştirir; ancak bu kazancın bir kısmı rutin ve soruşturma bağlamlarının ayrışmasından gelir ve numune türleri ayrı incelendiğinde küçülür."),
  bul("**Hipotez 3:** Küçük veride önceden eğitilmiş tablo modeli (TabPFN), sıfırdan eğitilen sinir ağlarından ve dizi modellerinden daha kararlı sonuç ve daha iyi kalibre olasılık üretir."),

  h2("1.4. Amaç ve Hedefler"),
  p("Araştırmanın temel amacı, içme suyu numunelerinde koliform uygunsuzluğunu laboratuvar sonucunu beklemeden ve modelin hiç görmediği bölgelerde tahmin eden yaklaşımları dürüst biçimde değerlendirmek; hangi bilginin ve hangi model türünün hangi koşulda katkı sağladığını belirlemek ve bunu su idaresinin kullanabileceği bir risk sıralaması prototipine dönüştürmektir. Çalışmanın hedefi laboratuvar analizinin yerini almak değil, numune önceliklendirmesini destekleyen bir araştırma prototipi ve nesnel yöntem karşılaştırmasıdır."),
  bul("1. Veri setinin sürümünü sabitlemek; etiket düzeltme kuralını, numune birleştirme adımlarını ve sızıntı denetimlerini kodla belgelemek; ZAPE, belediye, yıl ve numune türüne göre numune ve uygunsuz sayılarını raporlamak."),
  bul("2. Görülmemiş bölge, gelecek yıl ve görülmemiş belediye senaryolarını, eğitim ve testte ortak ZAPE veya ortak belediye-gün bulunmayacak biçimde sabitlemek."),
  bul("3. Lojistik Regresyon, LightGBM, TabPFN, MLP, 1D-CNN, GroupDRO ve CORAL modellerini aynı özellikler ve aynı bölmelerle eğitimsiz klor sıralamasına karşı karşılaştırmak."),
  bul("4. Bölge geçmişini el yapımı özetler ve dizi modelleri (GRU, 1D-CNN) ile kullanmak; geçmişin yalnızca rutin numunelerden hesaplandığı denetim deneyiyle kazancın kaynağını ayrıştırmak."),
  bul("5. Başarıyı PR-AUC, ROC-AUC, MCC, kesinlik, duyarlılık ve Brier skoru ile ölçmek; farkları ZAPE düzeyinde bootstrap güven aralıklarıyla test etmek; sonuçları rutin ve soruşturma numunelerinde ayrı raporlamak."),
  bul("6. Seçilen modelin olasılıklarının kalibrasyonunu, sabit alarm bütçesinde yakaladığı uygunsuz numune sayısını ve bölgeden bilinen ölçüm sayısına göre başarı eğrisini çıkarmak; kaçırılan uygunsuzluklar ve yanlış alarmlar üzerinden hata analizi yapmak."),
  bul("7. Bölgeleri tahmin edilen riske göre sıralayan bir demo uygulaması ve tüm deneylerin tek komutla yeniden üretilebildiği açık kaynak kod deposuyla sonuç raporunu tamamlamak."),

  h1("2. YÖNTEM"),
  h2("2.1. Ön Çalışma ve Fizibilite"),
  p("Önerilen araştırma, ekibin yürüttüğü ön çalışmaların kontrollü ve tekrarlanabilir biçimde genişletilmesidir. Ön çalışmalarda (i) yayımlanmış veri setinde laboratuvarın \"<\" işaretiyle verdiği \"tespit edilmedi\" sonuçlarının pozitif sayıldığı bir etiket hatası bulunmuş ve düzeltilmiş, (ii) aynı belediyede aynı gün alınan numunelerin aynı hava değerlerini paylaşmasından kaynaklanan bir veri sızıntısı tespit edilmiş, (iii) numaralandırılmış 17 deney betiğiyle modeller üç test senaryosunda karşılaştırılmıştır (Tablo 1, Şekil 1). Sızıntı engellendiğinde Rastgele Orman'ın PR-AUC değeri 0,42'den 0,27'ye düşmüştür; bu, sızıntının sonuçları ne ölçüde iyimser gösterebileceğini somut olarak ortaya koymaktadır."),
  cap("**Tablo 1.** Ön çalışma sonuçları: görülmemiş bölge testi (ZAPE gruplu, arındırılmış 5 katlı çapraz doğrulama, 3 tohum ortalaması). Değerler ROC-AUC / PR-AUC'dir."),
  table(["Model", "Kullanılan bilgi", "Tüm numuneler (522; 82 uygunsuz)", "Rutin numuneler (390; 25 uygunsuz)"], [
    ["Eğitimsiz klor sıralaması", "Yalnız serbest klor", "0,720 / 0,281", "0,848 / 0,270"],
    ["Lojistik Regresyon", "Klor, hava, ay", "0,754 / 0,347", "0,782 / 0,196"],
    ["1D-CNN (tablo, kendi yazdığımız)", "Klor, hava, ay", "0,686 / 0,276", "0,731 / 0,177"],
    ["1D-CNN (yayımlanmış orijinal mimari)", "Klor, hava, ay", "0,684 / 0,267", "0,722 / 0,179"],
    ["TabPFN", "Klor, hava, ay", "0,801 / 0,422", "0,752 / 0,132"],
    ["Lojistik Regresyon", "+ bölge geçmişi özeti", "0,825 / 0,519", "0,748 / 0,380"],
    ["1D-CNN (yayımlanmış orijinal mimari)", "+ bölge geçmişi özeti", "0,839 / 0,529", "0,746 / 0,322"],
    ["GRU (dizi, ayarı eğitimde seçilmiş)", "+ bölge geçmişi dizisi", "0,822 / 0,493", "0,684 / 0,266"],
    ["TabPFN", "+ bölge geçmişi özeti", "0,878 / 0,563", "0,814 / 0,328"],
  ], [2500, 1900, 2420, 2421], { right: [2, 3] }),
  sp(),
  img("s1_on_calisma.png", 600, 254),
  figcap("**Şekil 1.** Ön çalışmada modellerin görülmemiş bölge testindeki (a) ROC-AUC ve (b) PR-AUC değerleri. Mavi çubuklar tüm numuneleri, turuncu çubuklar rutin numuneleri gösterir. Noktalı çizgiler rastgele bir modelin düzeyidir (PR-AUC'de pozitif oranı: tüm numunelerde 0,157, rutin numunelerde 0,064)."),
  p("Kendi yazdığımız 1D-CNN'in düşük sonucunun bir uygulama hatasından kaynaklanıp kaynaklanmadığını sınamak için tablo verisi için yayımlanmış bir 1D-CNN (baosenguo, 2020) orijinal mimarisi ve eğitim ayarlarıyla aynı protokolde çalıştırılmış ve neredeyse aynı sonuç alınmıştır (ROC-AUC 0,684); bölge geçmişi eklendiğinde bu model 0,839'a yükselmiştir. Bu, tek başına klor, hava ve takvim bilgisinin evrişimli ağlar için yetersiz olduğunu, sinir ağlarının kazancının bölge geçmişinden geldiğini göstermektedir. Bu sonuçlar tek başına kesin bir üstünlük olarak yorumlanamaz. Tüm numunelerde TabPFN + bölge geçmişi klor sıralamasından anlamlı biçimde iyidir (PR-AUC farkı +0,285; %95 güven aralığı +0,121 ile +0,426); ancak rutin numunelerde hiçbir model klor sıralamasını anlamlı biçimde geçememiştir. Geçmiş özetleri yalnızca önceki rutin numunelerden hesaplandığında Lojistik Regresyon'daki kazancın büyük kısmı kaybolmuş (PR-AUC 0,519'dan 0,380'e), numune türleri ayrı incelendiğinde geçmiş bilgisi anlamlı kazanç sağlamamıştır. Ayarı eğitimde seçilen GRU ise gelecek yıl testinde rutin numunelerde rastgele düzeye düşmüştür (ROC-AUC 0,49). Bu nedenle önerilen projede kazancın kaynağı numune türüne, bölgeden bilinen ölçüm sayısına ve zamana göre ayrıştırılacak, sonuçların belirsizliği açıkça raporlanacaktır."),

  h2("2.2. Araştırma İş Akışı"),
  p("Çalışmanın genel akışı Şekil 2'de özetlenmiştir. Veri sürümü, etiket kuralı ve test bölmeleri sabitlendikten sonra bütün modeller aynı özellikler, aynı bölmeler ve aynı rastgele tohumlarla eğitilecek; yalnızca model türü ve kullanılan bilgi (klor/hava/takvim, bölge geçmişi özeti, bölge geçmişi dizisi) değiştirilecektir. Her deney numaralandırılmış ayrı bir betik olarak yazılacak ve tek komutla yeniden üretilebilecektir."),
  img("s2_is_akisi.png", 600, 216),
  figcap("**Şekil 2.** Önerilen araştırmanın iş akışı ve iş paketleriyle (İP) ilişkisi. Kesikli kutu, yayın sürecine bağlı koşullu çıktıyı göstermektedir."),

  h2("2.3. Veri Seti"),
  p("Araştırmanın temel veri kaynağı, Mendeley Data platformunda açık erişimli yayımlanan Tech4Aqua veri setinin 1. sürümüdür (de Castro Paula vd., 2026). Etiket düzeltmesinden sonra veri 522 numune, 82 uygunsuz numune ve 72 ZAPE içermektedir. Chaves'ten 354 numune (69 ZAPE, 67 uygunsuz), Montalegre'den 102 numune (2 ZAPE, 15 uygunsuz), Boticas'tan 66 numune (1 ZAPE, uygunsuz yok) bulunmaktadır. Rutin numuneler yıllık 18-24 numune ile 7 ZAPE'den alınmıştır; soruşturma numuneleri 2015'ten itibaren 66 ZAPE'ye yayılmaktadır ve 2025 yılındaki 38 numunenin tamamı soruşturma numunesidir (28 uygunsuz). Değişkenler Tablo 2'de, veri dağılımı Şekil 3'te verilmiştir. Günlük hava verisinin gecikmeli özellikleri Open-Meteo üzerinden ERA5 yeniden analiz verisinden (Hersbach vd., 2020) alınmaktadır. Veri setinin sonradan yayımlanan 2. sürümü, sürüm farkları ve yeni kayıtlar denetlendikten sonra ek zamansal test için değerlendirilecektir."),
  cap("**Tablo 2.** Bağımlı ve bağımsız değişkenler."),
  table(["Değişken", "Tür ve birim", "Açıklama"], [
    ["Koliform uygunsuzluğu (bağımlı)", "İkili (0/1)", "Düzeltilmiş koliform sonucu 0'dan büyükse 1; \"<\" işaretli (tespit edilmedi) sonuçlar 0."],
    ["Serbest klor kalıntısı", "Sürekli, mg/L", "Numune anında ölçülür; 0,04-4,2 (medyan 0,30); 15 numunede eksik."],
    ["Günlük en yüksek sıcaklık", "Sürekli, °C", "Numune günü; 2,2-37,7."],
    ["Günlük yağış", "Sürekli, mm", "Numune günü; 0-56,9."],
    ["Ortalama bağıl nem", "Sürekli, %", "Numune günü; 28,6-98,2."],
    ["Ay", "Kesikli, 1-12", "Mevsimsellik için."],
    ["Gecikmeli hava özellikleri", "Sürekli", "Numune gününden önceki günlerden: son 3, 7 ve 14 günün toplam yağışı; son 7 günde 1 mm üzeri yağışlı gün sayısı; son yağıştan bu yana geçen gün (en fazla 60); son 7 günün ortalama sıcaklığı."],
    ["Bölge geçmişi özetleri", "Sürekli / ikili", "Aynı ZAPE'nin sonucu bilinen önceki numunelerinden hesaplanır (Bölüm 2.4)."],
    ["ZAPE, belediye, tarih, numune türü", "Yalnızca analiz", "Model girdisi değildir; veri bölme, alt grup analizi ve raporlama için kullanılır."],
  ], [2500, 1500, 5241]),
  sp(),
  img("s3_veri.png", 600, 188),
  figcap("**Şekil 3.** Veri setinin yapısı: (a) yıllara göre rutin ve soruşturma numune sayıları ile uygunsuz numune sayısı, (b) uygun ve uygunsuz numunelerde serbest klor dağılımı, (c) ZAPE başına numune sayısı (logaritmik eksen). Numunelerin büyük kısmı birkaç ZAPE'de toplanırken 33 ZAPE'de yalnızca bir numune bulunmaktadır."),

  h2("2.4. Veri Hazırlama ve Özellikler"),
  p("Ham laboratuvar kayıtlarındaki koliform sonucu, ölçüm işaretiyle (\"<\", \"=\", \">\") birlikte okunacak; \"<\" işaretli sonuçlar 0 kabul edilecektir. 0 ile 1 arasında kesirli sonucu olan 5 numune için ayrıca duyarlılık analizi yapılacaktır. Aynı ZAPE'de aynı gün alınmış birden fazla kayıt tek numunede birleştirilecektir. Adında hedefle ilişkili ifadeler (coli, bact, falha, microbio) geçen sütunların model girdisine girmesi kodla engellenecek; her bölmede eğitim ve testte ortak ZAPE veya ortak belediye-gün bulunmadığı otomatik olarak doğrulanacaktır. Eksik değerler ve standartlaştırma parametreleri yalnızca eğitim katından öğrenilecektir."),
  p("**Bölge geçmişi.** Bir numune için bölge geçmişi, aynı ZAPE'de numune gününden en az 2 gün önce alınmış numunelerden oluşur; böylece laboratuvar sonucu henüz gelmemiş numuneler kullanılmaz. Bu kümedeki numune sayısı n, uygunsuz numune sayısı k olmak üzere şu özellikler hesaplanır: bilinen numune sayısı (n); yumuşatılmış uygunsuzluk oranı (k + 0,1) / (n + 1); son bilinen numunenin uygunsuz olup olmadığı; son bilinen numuneden ve son uygunsuz numuneden bu yana geçen gün (en fazla 730); geçmiş numunelerin ortalama klor değeri. Kazancın soruşturma bağlamından etkilenip etkilenmediğini ölçmek için aynı özellikler yalnızca önceki rutin numunelerden de hesaplanacaktır. Dizi modellerinde ise bölgenin son L bilinen numunesi (L ∈ {4, 8, 16}) zaman sırasıyla verilir; her adımda standartlaştırılmış klor, hava ve ay değerleri, sonuç (0/1) ve bugüne uzaklık log(1 + gün)/5 olarak kullanılır."),

  h2("2.5. Model ve Eğitim Stratejisi"),
  p("Küçük veride aşırı iyimserliği önlemek için tablo modellerinde geniş hiperparametre araması yapılmayacak, ayarlar önceden sabitlenecektir (Tablo 3). Dizi modellerinde dizi uzunluğu {4, 8, 16} ve eğitim turu {100, 200} kümelerinden, yalnızca eğitim verisi içinde yapılan ZAPE gruplu 3 katlı iç doğrulamayla PR-AUC'ye göre seçilecektir. Sınıf dengesizliği sınıf ağırlıklandırmasıyla ele alınacaktır. Derin modeller PyTorch (Paszke vd., 2019), diğerleri scikit-learn (Pedregosa vd., 2011) ve LightGBM (Ke vd., 2017) ile uygulanacaktır. Karar eşiği test verisine bakılmadan, eğitim katı içinde ZAPE gruplu iç doğrulamayla MCC'yi en büyüten değer olarak seçilecektir."),
  cap("**Tablo 3.** Karşılaştırılacak modeller ve ayarları."),
  table(["Model", "Tür", "Ayarlar"], [
    ["Eğitimsiz klor sıralaması", "Referans", "Risk = −(serbest klor); eksik değerler medyanla doldurulur; eğitim gerektirmez."],
    ["Lojistik Regresyon", "Doğrusal", "liblinear çözücü, dengeli sınıf ağırlığı, medyan doldurma ve standartlaştırma."],
    ["LightGBM", "Gradyan artırma (Ke vd., 2017)", "200 ağaç, öğrenme oranı 0,05, 7 yaprak, derinlik en fazla 4, yaprakta en az 15 örnek, L2 = 1."],
    ["TabPFN v2", "Önceden eğitilmiş transformer (Hollmann vd., 2025)", "Ağırlıklar güncellenmez; eğitim katı bağlam olarak verilir; 3 tohum."],
    ["MLP", "Derin, tam bağlantılı", "Gizli katmanlar 32 ve 16 nöron, ReLU, dropout 0,2; Adam (öğrenme oranı 0,001, ağırlık azaltma 0,001), 300 tur."],
    ["1D-CNN (tablo)", "Derin, evrişimli", "Özellik vektörü tek kanallı sinyal: iki Conv1d (16 kanal, çekirdek 3) ve 32 nöronlu katman."],
    ["1D-CNN (orijinal, soft-ordering)", "Derin, evrişimli (baosenguo, 2020)", "Kaggle MoA 2. sıra çözümünün mimarisi ve eğitim ayarları birebir: 4096 boyutlu katman, 256 kanallı sinyale dönüştürme, 3 evrişim bloğu; Adam, OneCycle (en yüksek 0,01), 25 tur, batch 128."],
    ["GroupDRO / CORAL", "Alan genelleme (Sagawa vd., 2020; Sun ve Saenko, 2016)", "MLP ile aynı ağ; belediye × sınıf grup ağırlıkları (η = 0,1) / belediyeler arası kovaryans cezası (ağırlık 1)."],
    ["GRU / 1D-CNN (dizi)", "Derin, yinelemeli / evrişimli (Cho vd., 2014)", "16 birimli GRU veya iki Conv1d; dizi özeti güncel özelliklerle birleştirilip 32 nöronlu katmana verilir; Adam (0,003)."],
  ], [2300, 2500, 4441]),
  sp(),

  h2("2.6. Değerlendirme Ölçütleri"),
  p("**Test senaryoları.** (1) Görülmemiş bölge: ZAPE gruplu, sınıf oranı korunan 5 katlı çapraz doğrulama; test numunesiyle aynı belediyede aynı gün alınmış numuneler eğitimden çıkarılır (arındırma); 3 tohumla tekrarlanır. (2) Gelecek yıl: 2015-2025 arasındaki her test yılı için model yalnızca önceki yıllarla eğitilir; eğitimde en az 5 uygunsuz numune ve 5 ZAPE yoksa o yıl atlanır. (3) Görülmemiş belediye: her seferinde bir belediye test olarak ayrılır."),
  p("**Ölçütler.** Uygunsuz numuneler azınlıkta olduğundan ana ölçüt kesinlik-duyarlılık eğrisi altındaki alandır (PR-AUC; Saito ve Rehmsmeier, 2015). Yanında ROC-AUC, MCC (Chicco ve Jurman, 2020), kesinlik, duyarlılık ve Brier skoru (Brier, 1950) raporlanacaktır. İki model arasındaki fark ZAPE düzeyinde eşleştirilmiş bootstrap ile test edilecektir (Efron ve Tibshirani, 1993): ZAPE'ler yerine koyarak 2000 kez yeniden örneklenecek, %95 güven aralığı 0'ı içermiyorsa fark anlamlı kabul edilecektir. Bütün sonuçlar tüm numuneler, rutin numuneler ve soruşturma numuneleri için ayrı ayrı verilecektir."),
  p("**Karar desteği analizleri.** Olasılıkların kalibrasyonu güvenilirlik eğrisi ve beklenen kalibrasyon hatası (ECE; Guo vd., 2017) ile ölçülecek, gerekirse ZAPE gruplu çapraz uydurmalı Platt düzeltmesi (Platt, 2000) uygulanacaktır. Alarm bütçesi analizinde su idaresinin numunelerin yalnızca %5, %10 veya %20'sine alarm verebildiği varsayılarak yakalanan uygunsuz numune sayısı hesaplanacaktır. Başarı eğrisi, başarımı bölgeden sonucu bilinen geçmiş numune sayısına göre gösterecektir (Şekil 4). Hata analizinde kaçırılan uygunsuzluklar ve yanlış alarmlar numune türü, belediye, klor düzeyi, mevsim ve geçmiş uzunluğuna göre incelenecektir."),
  img("s4_basari_egrisi.png", 520, 245),
  figcap("**Şekil 4.** Ön çalışmada bölgeden sonucu bilinen önceki numune sayısına göre ROC-AUC (parantez içinde numune ve uygunsuz sayısı). Bölgeden 1-5 sonuç bilindiğinde geçmiş kullanan modeller klorun açıkça önündedir; uzun süredir izlenen bölgelerde (11 ve üzeri; çoğu rutin) hiçbir model klor sıralamasına ulaşamamaktadır. Gruplardaki uygunsuz sayısı az olduğundan değerler belirsizdir."),

  h2("2.7. Başarı Ölçütleri ve Riskler"),
  p("Projenin başarısı tek bir başarım değerine değil, aşağıdaki doğrulanabilir koşullara bağlanacaktır: (i) üç test senaryosunun kodla denetlenen ve tek komutla yeniden üretilen biçimde çalışması; (ii) en az yedi modelin eğitimsiz klor sıralamasına karşı ZAPE düzeyinde güven aralıklarıyla, numune türlerine ayrılmış olarak karşılaştırılması; (iii) bölge geçmişinin katkısının rutin ve soruşturma bağlamına göre ayrıştırılması; (iv) seçilen modelin kalibrasyon, alarm bütçesi ve hata analizinin raporlanması; (v) bölgeleri risk sırasına dizen demo uygulamanın çalışır durumda gösterilmesi. Olumsuz sonuçlar (örneğin rutin numunelerde klor sıralamasının geçilememesi) da bulgu olarak raporlanacaktır."),
  table(["Risk", "Olasılık / etki", "Önlem veya B planı"], [
    ["Rutin numunelerde uygunsuz sayısının az olması (25) nedeniyle farkların anlamlı çıkmaması", "Yüksek / yüksek", "Bölge düzeyinde bootstrap ve alarm bütçesi analiziyle belirsizliği açıkça raporlamak; rutin erken uyarı üzerine kesin iddia kurmamak."],
    ["Bölge geçmişi kazancının numune türü ayrımından gelmesi", "Yüksek / orta", "Geçmişin yalnız rutin numunelerden hesaplandığı denetim deneyini ana analizlere eklemek; sonuçları numune türüne göre ayrı vermek."],
    ["Veride yeni bir sızıntı veya etiket sorunu bulunması", "Orta / yüksek", "Denetimleri koda eklemek; tüm deneyleri tek komutla yeniden çalıştırmak; değişiklikleri sürüm notunda belgelemek."],
    ["Dizi modellerinin zaman değişince kararsız kalması", "Orta / orta", "Ayarları yalnız eğitim verisinde seçmek; hibrit modeli (dizi + özet) denemek; kararlı model olarak TabPFN veya LR'yi kullanmak."],
    ["Ek veri sürümünün farklı yapıda olması", "Orta / düşük", "Ana veri setiyle araştırmayı sürdürmek; ek sürümü yalnızca sürüm farkı denetiminden sonra ayrı test olarak kullanmak."],
    ["Hesaplama ve arayüz sorunları", "Düşük / orta", "TabPFN ve küçük ağları işlemci üzerinde çalıştırmak; hafif bir web arayüzü veya komut satırı sürümü kullanmak."],
  ], [3000, 1300, 4941]),
  sp(),

  h1("3. PROJE YÖNETİMİ"),
  h2("3.1. İş-Zaman Çizelgesi"),
  p("Aşağıdaki 8 aylık plan destek başlangıcından itibaren önerilen çalışma takvimidir. Ön çalışmalarda tamamlanan etiket düzeltmesi, ilk model karşılaştırmaları ve deneme analizleri burada yeniden yapılacak faaliyet olarak sayılmamıştır; bu adımlar sabit protokollerle doğrulanacak ve genişletilecektir."),
  table(["İş paketi", "Planlanan faaliyet", "Dönem", "Doğrulanabilir çıktı", "Başarı (%)"], [
    ["İP1", "Veri sürümü, etiket ve sızıntı denetimi; üç test protokolünün sabitlenmesi", "1-2. ay", "Veri denetim raporu, sabit bölme dosyaları, otomatik denetim testleri", "15"],
    ["İP2", "Tablo, önceden eğitilmiş ve derin modellerin üç senaryoda karşılaştırılması", "2-3. ay", "Eş koşullu deney tabloları, bootstrap güven aralıkları", "15"],
    ["İP3", "Bölge geçmişi özetleri ve dizi modelleri; rutin/soruşturma ayrıştırması", "3-5. ay", "Geçmiş katkısı tabloları, başarı eğrisi", "25"],
    ["İP4", "Zamansal ve belediyeler arası testler, kalibrasyon, alarm bütçesi, hata analizi", "5-6. ay", "Kalibrasyon grafikleri, alarm bütçesi tablosu, hata dosyası", "20"],
    ["İP5", "Veri setinin 2. sürümüyle ek zamansal test (sürüm farkı denetiminden sonra)", "6-7. ay", "Sürüm karşılaştırma notu, ek test sonuçları", "10"],
    ["İP6", "Erken uyarı demo uygulaması ve sonuç raporu", "7-8. ay", "Çalışan demo, açık kod deposu, sonuç raporu", "15"],
    ["Toplam", "", "", "", "100"],
  ], [900, 3300, 900, 3241, 900], { right: [4] }),
  sp(),
  h2("3.2. Araştırma Olanakları"),
  p("Proje kapsamında veri işleme, model eğitimi ve değerlendirme çalışmaları için araştırma ekibinin mevcut bilgisayar ve yazılım olanaklarından yararlanılacaktır: [Afra Ordulu: bilgisayar modeli, işlemci, bellek] ve [Nilüfer Öden: bilgisayar modeli, işlemci, bellek]. Ön çalışmalardaki tüm deneyler, TabPFN dâhil, grafik işlemci gerektirmeden işlemci üzerinde çalıştırılmıştır; veri setinin küçük olması nedeniyle ek donanım gereksinimi öngörülmemektedir."),
  p("Modellerin geliştirilmesinde Python programlama dili ile scikit-learn, LightGBM, PyTorch ve TabPFN açık kaynak kütüphanelerinden yararlanılacaktır. Kod, ara sonuçlar ve deney betikleri GitHub üzerinde sürüm kontrolüyle tutulmaktadır; ekibin mevcut veri hazırlama ve deney kodları projenin teknik altyapısını oluşturmaktadır. Daha yüksek hesaplama gücü gerekmesi durumunda ücretsiz veya kurumsal bulut kaynakları değerlendirilecektir. Bütçe planlaması, araştırmanın ihtiyaçları ve TÜBİTAK 2209-A programının destek koşulları doğrultusunda yapılacaktır."),

  h1("4. YAYGIN ETKİ"),
  p("Çalışmanın beklenen bilimsel katkısı, içme suyunda koliform uygunsuzluğu tahmininde hangi bilginin (klor, hava, bölge geçmişi) ve hangi model türünün hangi koşulda katkı sağladığını sızıntısız bir deney protokolüyle raporlamasıdır. Yayımlanmış açık veri setindeki etiket hatası ve sızıntı bulguları, aynı veriyi kullanacak diğer araştırmacılar için de yararlıdır. Rutin numunelerde basit klor sıralamasının güçlü kaldığı gibi olumsuz bulgular ve küçük örnekli gruplara ilişkin sınırlılıklar açıkça belgelenecektir. Uygun koşullar oluşursa ulusal bir kongrede bildiri veya makale hazırlanması hedeflenmektedir; yayın kabulü garanti edilmemektedir."),
  p("Toplumsal açıdan çalışma, su idarelerinin sınırlı numune ve laboratuvar kapasitesini riskli bölgelere yönlendirmesine yardımcı olabilecek bir prototip sunmayı amaçlamaktadır. Karar yetkisi su idaresinde ve laboratuvar analizinde kalacaktır. Yöntem, uygun veriye erişilmesi durumunda Türkiye'deki içme suyu izleme verilerine uyarlanabilir. Öğrenci araştırmacılar açısından makine öğrenmesi, derin öğrenme, güvenilir yapay zekâ değerlendirmesi, veri yönetimi ve bilimsel raporlama becerilerinin gelişmesi beklenmektedir."),

  h1("5. BEKLENEN ÇIKTILAR"),
  table(["Çıktı", "Açıklama / ölçülebilir gösterge"], [
    ["Veri denetim raporu", "Etiket düzeltme kuralı, sızıntı denetimleri, ZAPE / belediye / yıl / numune türüne göre numune ve uygunsuz sayıları"],
    ["Sızıntısız değerlendirme çerçevesi", "Üç test senaryosu, kodla doğrulanan bölmeler, tek komutla yeniden üretilen deney betikleri"],
    ["Model karşılaştırması", "En az yedi modelin klor sıralamasına karşı ZAPE düzeyinde güven aralıklı sonuçları; rutin ve soruşturma numunelerinde ayrı"],
    ["Bölge geçmişi analizi", "Geçmiş özetleri ve dizi modellerinin katkısı, yalnız rutin geçmiş denetimi, başarı eğrisi"],
    ["Karar desteği analizleri", "Kalibrasyon grafikleri, sabit alarm bütçesi tabloları, kaçırılan uygunsuzluk ve yanlış alarm örnekleri"],
    ["Demo uygulaması", "Bölgeleri tahmin edilen riske göre sıralayan, yeni ölçüm girilebilen çalışan prototip"],
    ["Sonuç raporu", "Kaynakça, yöntem sürümleri, sınırlılıklar ve yeniden üretilebilirlik notları; açık kod deposu"],
  ], [2800, 6441], { labelCol: true }),
  sp(),

  h1("6. KAYNAKLAR"),
  ...REFS.map(t => new Paragraph({ children: runs(t, { size: SMALL + 1 }), alignment: AlignmentType.LEFT,
    indent: { left: 360, hanging: 360 }, spacing: { after: 70 } })),

  h1("7. EKLER"),
  p("Ek A - Ön çalışma deney tabloları, kalibrasyon ve alarm bütçesi grafikleri (başvuru sisteminin ek yükleme koşullarına bağlı olarak eklenecektir)."),
];

const doc = new Document({
  styles: { default: { document: { run: { font: FONT, size: SZ } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 26, bold: true, color: BLUE }, paragraph: { spacing: { before: 260, after: 120 }, keepNext: true, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 22, bold: true, color: "2E74B5" }, paragraph: { spacing: { before: 200, after: 90 }, keepNext: true, outlineLevel: 1 } },
    ] },
  numbering: { config: [{ reference: "b", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
    style: { paragraph: { indent: { left: 400, hanging: 240 } } } }] }] },
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1219, bottom: 1134, left: 1361, right: 1304 } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
      children: [new TextRun({ text: "TÜBİTAK 2209-A | Araştırma Önerisi", font: FONT, size: 16, color: "595959" })] })] }) },
    children }],
});
Packer.toBuffer(doc).then(b => { fs.writeFileSync(OUTFILE, b); console.log("ok", REFS.length); });
