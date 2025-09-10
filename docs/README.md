# Lost Baggage System (LBS) - Proje Dokümantasyonu

## İçindekiler
1. [Proje Genel Bakışı](#proje-genel-bakışı)
2. [Teknik Mimari](#teknik-mimari)
3. [Kurulum ve Yapılandırma](#kurulum-ve-yapılandırma)
4. [Kullanım Kılavuzu](#kullanım-kılavuzu)
5. [API Referansı](#api-referansı)
6. [Örnekler](#örnekler)

## Proje Genel Bakışı

Lost Baggage System (LBS), havaalanları ve güvenlik birimleri için geliştirilmiş, yapay zeka destekli bir bagaj tanımlama ve eşleştirme sistemidir.

### Ana Özellikler
- **Çok Boyutlu Analiz**: Renk, şekil, doku ve görsel özellik analizi
- **Akıllı Arama**: Benzerlik tabanlı eşleştirme algoritmaları
- **Ölçeklenebilir Mimari**: Modüler ve genişletilebilir yapı
- **Toplu İşleme**: Batch processing desteği

## Teknik Mimari

### Sistem Bileşenleri

#### 1. Analiz Motoru (`src/core/`)
- **LuggageAnalyzer**: Ana koordinatör sınıf
- Çok boyutlu özellik çıkarımı
- Güven skorları hesaplama

#### 2. Alt Analizörler (`src/analyzers/`)
- **ColorAnalyzer**: Renk ve malzeme analizi
- **ShapeAnalyzer**: Şekil ve boyut tespiti
- **TextureAnalyzer**: Doku ve yüzey analizi

#### 3. Arama Motoru (`src/search/`)
- **LuggageSearcher**: Ana arama koordinatörü
- **SimilarityCalculator**: Benzerlik hesaplama
- **GroupManager**: Küme tabanlı optimizasyon

#### 4. Yardımcı Modüller (`src/utils/`)
- **ImageProcessor**: Resim ön işleme
- **FeatureExtractor**: Görsel özellik çıkarımı

## Kurulum ve Yapılandırma

### Sistem Gereksinimleri
- Python 3.8+
- 4GB+ RAM
- OpenCV compatible sistem

### Kurulum Adımları

```bash
# Repository klonla
git clone https://github.com/your-repo/lbs5.git
cd lbs5

# Sanal ortam oluştur
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\\Scripts\\activate  # Windows

# Bağımlılıkları yükle
pip install -r requirements.txt
```

### Konfigürasyon

Sistem ayarları `config/default_config.json` dosyasında tanımlıdır:

```json
{
  "analysis": {
    "color": {
      "num_clusters": 3,
      "material_detection": true
    },
    "shape": {
      "wheel_detection_enabled": true
    }
  }
}
```

## Kullanım Kılavuzu

### 1. Tek Resim Analizi

```bash
python main.py analyze --image bagaj.jpg --output sonuc.json --mode normal
```

### 2. Toplu Analiz

```bash
python main.py batch --input resimler/ --output analizler/ --mode normal
```

### 3. Bagaj Arama

```bash
python main.py search --query sorgu.json --gallery galeri/ --output sonuc.json --top-k 10
```

### Analiz Modları
- **lite**: Hızlı analiz (12 SIFT keypoint)
- **normal**: Standart analiz (20 SIFT keypoint)
- **detailed**: Detaylı analiz (50 SIFT keypoint)

### Arama Modları
- **global**: Tüm galeri taraması
- **group_based**: Küme tabanlı hızlı arama
- **multi_image**: Çoklu sorgu desteği

## API Referansı

### LuggageAnalyzer

```python
from src.core.luggage_analyzer import LuggageAnalyzer

analyzer = LuggageAnalyzer(config)
result = analyzer.analyze_image("bagaj.jpg", mode="normal")
```

#### Çıktı Formatı
```json
{
  "metadata": {
    "source_filename": "bagaj.jpg",
    "analysis_timestamp": "2025-09-10T14:30:45Z"
  },
  "color_material": {
    "dominant_colors": [...],
    "material_type": "plastic"
  },
  "form_size": {
    "suitcase_type": "hardshell",
    "size_class": "medium"
  },
  "visual_features": {
    "sift_keypoints": [...],
    "global_embedding": [...]
  }
}
```

### LuggageSearcher

```python
from src.search.luggage_searcher import LuggageSearcher

searcher = LuggageSearcher(config)
searcher.load_gallery("galeri_klasoru/")
results = searcher.search(query_data, mode="global", top_k=10)
```

## Örnekler

### Örnek 1: Basit Analiz
```python
import json
from src.core.luggage_analyzer import LuggageAnalyzer

# Analyzer oluştur
analyzer = LuggageAnalyzer()

# Resmi analiz et
result = analyzer.analyze_image("test_bagaj.jpg")

# Sonucu kaydet
with open("analiz_sonucu.json", "w") as f:
    json.dump(result, f, indent=2)

print(f"Bagaj tipi: {result['form_size']['suitcase_type']}")
print(f"Ana renk: {result['color_material']['dominant_colors'][0]['name']}")
```

### Örnek 2: Toplu Arama
```python
from src.search.luggage_searcher import LuggageSearcher

# Searcher oluştur ve galeri yükle
searcher = LuggageSearcher()
searcher.load_gallery("data/gallery/")

# Toplu arama yap
results = searcher.batch_search(
    query_folder="data/queries/",
    output_folder="data/results/",
    search_mode="group_based",
    top_k=5
)

print(f"İşlenen sorgu sayısı: {results['total_queries']}")
print(f"Başarılı arama: {results['successful_searches']}")
```

## Performans Optimizasyonu

### Bellek Yönetimi
- Büyük resimler otomatik olarak yeniden boyutlandırılır
- Batch işleme sırasında bellek temizliği yapılır

### Hız Optimizasyonu
- Grup tabanlı arama kullanın (`group_based` mode)
- Lite mode düşük kaliteli resimler için yeterli
- Paralel işleme konfigürasyonda etkinleştirilebilir

## Hata Yönetimi

Sistem, her seviyede hata yakalama mekanizmaları içerir:
- Desteklenmeyen format uyarıları
- Bozuk resim dosyaları atlanır
- JSON şema validasyonu
- Güven skoru tabanlı filtreleme

## Katkıda Bulunma

1. Fork yapın
2. Feature branch oluşturun (`git checkout -b feature/yeni-ozellik`)
3. Commit yapın (`git commit -am 'Yeni özellik eklendi'`)
4. Push yapın (`git push origin feature/yeni-ozellik`)
5. Pull Request oluşturun

## Lisans

MIT License - Detaylar için `LICENSE` dosyasını inceleyin.
