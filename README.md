# Lost Baggage System (LBS)

## Proje Açıklaması
Lost Baggage System (LBS), havaalanları ve güvenlik birimleri için geliştirilmiş, yapay zeka destekli bir bagaj tanımlama ve eşleştirme sistemidir. Bu sistem, kayıp veya sahipsiz bagajların fotoğraflarını analiz ederek, önceden oluşturulmuş bir veri tabanı ile karşılaştırma yaparak eşleşmeleri bulur.

## Ana Özellikler
- **Renk Analizi**: K-means kümeleme ile dominant renk çıkarımı
- **Şekil ve Boyut Analizi**: Bagaj tipi, boyut sınıfı, tekerlek sayısı tespiti
- **Görsel Özellik Çıkarımı**: SIFT keypoints, texture patterns, global embedding
- **Aksesuar Tespiti**: Fermuar, kilit, dış cep analizi
- **Gelişmiş Arama**: Global search, group-based search, multi-image query
- **HEIC Desteği**: iPhone ve modern cihazlardan HEIC/HEIF formatı desteği
- **AI Maskeleme**: SAM (Segment Anything Model) ile otomatik bagaj tespiti ve arka plan kaldırma

## AI Maskeleme Sistemi
LBS5, görüntülerdeki bagajları otomatik tespit edip arka planı kaldıran gelişmiş bir maskeleme sistemi içerir:

### Teknolojiler
- **SAM (Segment Anything Model)**: Meta'nın en gelişmiş segmentasyon modeli
- **RemBG**: Alternatif arka plan kaldırma
- **OpenCV**: Kenar tabanlı fallback yöntem

### Kullanım
```bash
# Bagaj maskeleme
python scripts/mask_luggage.py -i data/input -o data/input-masked

# SAM model seçimi
python scripts/mask_luggage.py --model-type vit_h --device cuda

# Maskeleme istatistikleri
python scripts/mask_luggage.py --stats

# Çıktı klasörünü temizle
python scripts/mask_luggage.py --clear-output
```

### Özellikler
- **Otomatik Model Seçimi**: SAM → RemBG → Edge Detection fallback
- **HEIC/HEIF Desteği**: iPhone fotoğrafları için tam destek
- **GPU Hızlandırma**: CUDA destekli hızlı işleme
- **Batch İşleme**: Çoklu dosya işleme desteği
- **Kalite Kontrolü**: Ayarlanabilir çıktı kalitesi

## Teknoloji Stack
- **Python 3.8+**
- **OpenCV**: Görüntü işleme
- **scikit-learn**: Makine öğrenmesi
- **NumPy/Pandas**: Veri işleme
- **PyTorch**: AI model altyapısı
- **SAM**: Gelişmiş segmentasyon
- **JSON Schema**: Veri doğrulama

## Kurulum
```bash
pip install -r requirements.txt
```

## Kullanım
```bash
# Maskeleme sistemi
python scripts/mask_luggage.py

# Tek resim analizi (HEIC destekli)
python main.py analyze --image bagaj.heic --output sonuc.json

# Klasör bazlı arama
python main.py search --query sorgu.json --gallery galeri/ --top-k 10

# Batch işleme
python main.py batch --input data/queries/ --output data/results/
```

## İş Akışı
```
Orijinal Fotoğraf → AI Maskeleme → Özellik Analizi → Arama & Eşleştirme
```

1. **Maskeleme**: SAM ile bagaj tespiti ve arka plan kaldırma
2. **Analiz**: Renk, şekil, texture özellikleri çıkarımı  
3. **Arama**: Veri tabanında benzerlik araması
4. **Sonuç**: Eşleşen bagajlar ve güven skorları

## Önemli Notlar
- **Croplanmış Fotoğraflar**: Sistem croplanmış bagaj fotoğrafları ile çalışacak şekilde optimize edilmiştir
- **Tek Analiz Modu**: Sistem artık sadece en detaylı analiz modunu kullanır (50 SIFT keypoint)
- **HEIC Desteği**: iPhone ve modern cihazlardan gelen HEIC/HEIF formatları desteklenir
- **Malzeme Analizi**: Malzeme sınıflandırması kaldırılmış, sadece renk ve yüzey analizi yapılır
- **AI Maskeleme**: SAM ile profesyonel kalitede bagaj tespiti ve arka plan temizleme

## Proje Yapısı
```
lbs5/
├── src/
│   ├── core/           # Ana sistem bileşenleri
│   ├── analyzers/      # Analiz modülleri
│   ├── search/         # Arama algoritmaları
│   ├── mask/           # AI maskeleme sistemi
│   └── utils/          # Yardımcı fonksiyonlar
├── data/
│   ├── input/          # Orijinal bagaj fotoğrafları
│   ├── input-masked/   # Maskelenmiş fotoğraflar
│   ├── gallery/        # Referans bagaj fotoğrafları
│   ├── queries/        # Sorgu fotoğrafları
│   └── results/        # Analiz sonuçları
├── scripts/            # CLI araçları
│   └── mask_luggage.py # Maskeleme CLI
├── tests/              # Test dosyaları
├── config/             # Konfigürasyon dosyaları
└── docs/               # Dokümantasyon
```