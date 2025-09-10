 # LBS Maskeleme Sistemi Dokümantasyonu

## Genel Bakış
LBS5 maskeleme sistemi, bagaj fotoğraflarından arka planı otomatik olarak kaldıran gelişmiş bir AI sistemidir. SAM (Segment Anything Model) tabanlı olarak çalışır ve fallback yöntemler sunar.

## Sistem Mimarisi

### Katmanlı Yaklaşım
1. **SAM (Segment Anything Model)** - Birincil yöntem
2. **RemBG** - İkincil yöntem  
3. **Kenar Tabanlı Tespit** - Son çare yöntemi

### Ana Bileşenler

#### SAMDetector
- SAM model yönetimi
- GPU/CPU otomatik seçimi
- Model tipi seçimi (vit_h, vit_l, vit_b)
- Fallback mekanizması

#### MaskProcessor
- Batch işleme
- Dosya yönetimi
- Format desteği (HEIC/HEIF dahil)
- İstatistik toplama

## Kullanım Kılavuzu

### Kurulum
```bash
# Gerekli kütüphaneleri yükle
pip install -r requirements.txt

# SAM modelini manuel indirme (opsiyonel)
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

### Temel Kullanım
```bash
# Basit maskeleme
python scripts/mask_luggage.py

# Özel klasörler
python scripts/mask_luggage.py -i photos/ -o masked/

# GPU kullanımı
python scripts/mask_luggage.py --device cuda

# Yüksek kalite
python scripts/mask_luggage.py --quality 100 --model-type vit_h
```

### Gelişmiş Seçenekler
```bash
# Sadece istatistikleri göster
python scripts/mask_luggage.py --stats

# Çıktı klasörünü temizle
python scripts/mask_luggage.py --clear-output

# Maskeleri kaydetme
python scripts/mask_luggage.py --no-save-masks

# Var olan dosyaları üzerine yaz
python scripts/mask_luggage.py --overwrite

# Detaylı log
python scripts/mask_luggage.py -v
```

## Programatik Kullanım

### Basit Örnek
```python
from src.mask.mask_processor import MaskProcessor

# Processor oluştur
processor = MaskProcessor(
    input_dir="data/input",
    output_dir="data/input-masked"
)

# Tüm resimleri işle
results = processor.process_all_images()
print(f"Başarılı: {results['success_count']}")
```

### Özelleştirilmiş Konfigürasyon
```python
from src.mask.mask_processor import MaskProcessor

# SAM konfigürasyonu
sam_config = {
    "model_type": "vit_l",
    "device": "cuda",
    "confidence_threshold": 0.7
}

# Processor oluştur
processor = MaskProcessor(
    input_dir="photos/",
    output_dir="masked/",
    sam_config=sam_config
)

# Tek dosya işle
result = processor.process_single_image(
    image_path=Path("bagaj.jpg"),
    save_mask=True,
    quality=95
)
```

### SAM Detector Kullanımı
```python
from src.mask.sam_detector import SAMDetector
import cv2

# Detector oluştur
detector = SAMDetector(
    model_type="vit_h",
    device="cuda"
)

# Resmi yükle
image = cv2.imread("bagaj.jpg")

# Bagaj tespiti
mask = detector.detect_luggage(image)

if mask is not None:
    # Maskeyi uygula
    masked_image = image * (mask[:, :, None] / 255.0)
    cv2.imwrite("masked_bagaj.jpg", masked_image)
```

## Konfigürasyon

### config.json Maskeleme Bölümü
```json
{
  "masking": {
    "enabled": true,
    "sam_config": {
      "model_type": "vit_h",
      "device": "auto",
      "confidence_threshold": 0.5
    },
    "fallback_methods": {
      "rembg": true,
      "simple_edge": true
    },
    "output": {
      "save_masks": true,
      "quality": 95,
      "background_color": [255, 255, 255]
    }
  }
}
```

### Model Tipleri
- **vit_h**: En büyük ve en doğru model (2.5GB)
- **vit_l**: Orta boyut ve doğruluk (1.2GB) 
- **vit_b**: En küçük ve en hızlı model (375MB)

### Cihaz Seçenekleri
- **auto**: Otomatik CUDA/CPU seçimi
- **cuda**: GPU zorlaması
- **cpu**: CPU zorlaması

## Performans Optimizasyonu

### GPU Kullanımı
```bash
# CUDA kontrol
python -c "import torch; print(torch.cuda.is_available())"

# GPU bellek kontrolü
nvidia-smi
```

### Batch İşleme İpuçları
- Büyük dosyalar için `vit_b` modeli kullanın
- Paralel işleme için birden fazla instance çalıştırın
- Kalite/hız dengesini ayarlayın

### Bellek Yönetimi
```python
# Büyük dosyalar için model yeniden yükleme
processor = MaskProcessor(...)
for i, image_file in enumerate(image_files):
    if i % 100 == 0:  # Her 100 dosyada bir
        processor.sam_detector = SAMDetector(...)  # Model yenile
```

## Hata Ayıklama

### Yaygın Sorunlar

#### SAM Model İndirme Hatası
```bash
# Manuel indirme
mkdir -p ~/.cache/torch/hub/checkpoints/
wget -O ~/.cache/torch/hub/checkpoints/sam_vit_h_4b8939.pth \
     https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

#### CUDA Bellek Hatası
```python
# Model tipini değiştir
sam_config = {"model_type": "vit_b", "device": "cpu"}
```

#### HEIC Format Hatası
```bash
# HEIC desteği kurulumu
pip install pillow-heif
```

### Log Analizi
```bash
# Detaylı loglar
python scripts/mask_luggage.py -v 2>&1 | tee maskeleme.log

# Hata filtreleme
grep "ERROR" maskeleme.log
```

## Kalite Kontrolü

### Mask Kalitesi Değerlendirmesi
```python
def evaluate_mask_quality(original, mask):
    # Maske kapsamı
    coverage = np.sum(mask > 0) / mask.size
    
    # Kenar kalitesi
    edges = cv2.Canny(mask, 50, 150)
    edge_quality = np.sum(edges > 0) / edges.size
    
    return {
        "coverage": coverage,
        "edge_quality": edge_quality
    }
```

### Otomatik Kalite Kontrol
```python
def auto_quality_check(results):
    for result in results["processed"]:
        if result["success"]:
            # Dosya boyutu kontrolü
            output_size = Path(result["output_file"]).stat().st_size
            if output_size < 1000:  # 1KB'den küçükse
                print(f"Uyarı: {result['output_file']} çok küçük")
```

## Entegrasyon

### Ana LBS Sistemi ile Entegrasyon
```python
from src.core.luggage_analyzer import LuggageAnalyzer
from src.mask.mask_processor import MaskProcessor

# Önce maskeleme
processor = MaskProcessor("data/input", "data/input-masked")
mask_results = processor.process_all_images()

# Sonra analiz
analyzer = LuggageAnalyzer()
for result in mask_results["processed"]:
    if result["success"]:
        analysis = analyzer.analyze_image(result["output_file"])
```

### API Entegrasyonu
```python
from flask import Flask, request, jsonify
from src.mask.mask_processor import MaskProcessor

app = Flask(__name__)
processor = MaskProcessor()

@app.route('/mask', methods=['POST'])
def mask_image():
    file = request.files['image']
    # Geçici kaydet, işle, sonuç döndür
    # ...
```

## Sonuç
LBS maskeleme sistemi, modern AI teknolojilerini kullanarak bagaj tespiti ve arka plan kaldırma işlemlerini otomatikleştirir. SAM tabanlı yaklaşım, yüksek kaliteli sonuçlar sunarken, fallback mekanizmaları sistem güvenilirliğini artırır.
