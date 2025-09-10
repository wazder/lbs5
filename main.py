"""
Lost Baggage System (LBS) - Ana program
Command line arayüzü ve temel kullanım örnekleri
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Proje kök dizinini path'e ekle
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.core.luggage_analyzer import LuggageAnalyzer
from src.search.luggage_searcher import LuggageSearcher


def load_config(config_path: str = None) -> dict:
    """Konfigürasyon dosyasını yükler."""
    if config_path is None:
        config_path = project_root / "config" / "config.json"
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Konfigürasyon yükleme hatası: {e}")
        return {}


def analyze_single_image(image_path: str, output_path: str = None, 
                        config: dict = None, use_masking: bool = None):
    """Tek bir bagaj resmini analiz eder."""
    try:
        analyzer = LuggageAnalyzer(config.get('analysis', {}) if config else None)
        
        print(f"Analiz ediliyor: {image_path}")
        
        # Maskeleme durumunu göster
        if use_masking is None:
            mask_status = "Otomatik (config'e göre)"
        elif use_masking:
            mask_status = "Zorla Aktif"
        else:
            mask_status = "Devre Dışı"
        print(f"Maskeleme: {mask_status}")
        
        result = analyzer.analyze_image(image_path, use_masking=use_masking)
        
        if 'error' in result:
            print(f"Analiz hatası: {result['error']}")
            return False
        
        # Maskeleme kullanılıp kullanılmadığını göster
        if result.get('masking_used', False):
            print("SAM maskeleme kullanıldı")
        else:
            print("Orijinal resim kullanıldı")
        
        # SIFT özellik sayısını göster
        sift_count = len(result.get('visual_features', {}).get('sift_keypoints', []))
        print(f"🔍 SIFT keypoints: {sift_count}")
        
        # Sonucu kaydet
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"Sonuç kaydedildi: {output_path}")
        else:
            print(json.dumps(result, indent=2, ensure_ascii=False))
        
        return True
        
    except Exception as e:
        print(f"Analiz hatası: {str(e)}")
        return False


def analyze_batch(input_folder: str, output_folder: str, 
                 config: dict = None):
    """Klasördeki tüm resimleri toplu analiz eder."""
    try:
        analyzer = LuggageAnalyzer(config.get('analysis', {}) if config else None)
        
        # Desteklenen format (HEIC dahil)
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.heic', '.heif']
        
        # Input klasöründeki resimleri bul
        image_files = []
        for ext in image_extensions:
            image_files.extend(Path(input_folder).glob(f"*{ext}"))
            image_files.extend(Path(input_folder).glob(f"*{ext.upper()}"))
        
        if not image_files:
            print(f"'{input_folder}' klasöründe resim dosyası bulunamadı.")
            return False
        
        print(f"{len(image_files)} resim bulundu. Analiz başlatılıyor...")
        
        # Output klasörünü oluştur
        os.makedirs(output_folder, exist_ok=True)
        
        success_count = 0
        
        for image_file in image_files:
            try:
                print(f"İşleniyor: {image_file.name}")
                
                result = analyzer.analyze_image(str(image_file))
                
                if 'error' not in result:
                    # JSON dosyası olarak kaydet
                    output_file = Path(output_folder) / f"{image_file.stem}.json"
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(result, f, indent=2, ensure_ascii=False)
                    
                    success_count += 1
                else:
                    print(f"  Hata: {result['error']}")
            
            except Exception as e:
                print(f"  İşleme hatası: {str(e)}")
        
        print(f"Toplu analiz tamamlandı: {success_count}/{len(image_files)} başarılı")
        return True
        
    except Exception as e:
        print(f"Toplu analiz hatası: {str(e)}")
        return False


def search_luggage(query_path: str, gallery_path: str, output_path: str = None,
                  config: dict = None, top_k: int = 10, mode: str = 'global'):
    """Bagaj arama işlemi yapar."""
    try:
        searcher = LuggageSearcher(config.get('search', {}) if config else None)
        
        # Galeri yükle
        print(f"Galeri yükleniyor: {gallery_path}")
        if not searcher.load_gallery(gallery_path):
            print("Galeri yüklenemedi!")
            return False
        
        # Query verilerini yükle
        print(f"Sorgu yükleniyor: {query_path}")
        with open(query_path, 'r', encoding='utf-8') as f:
            query_data = json.load(f)
        
        # Arama yap
        print(f"Arama yapılıyor (mod: {mode}, top-{top_k})...")
        results = searcher.search(query_data, mode, top_k)
        
        # Sonuçları göster
        print(f"\n{len(results)} sonuç bulundu:")
        for i, result in enumerate(results, 1):
            print(f"{i}. {result['gallery_id']} - Skor: {result['similarity_score']:.4f}")
        
        # Sonucu kaydet
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'query_file': query_path,
                    'search_mode': mode,
                    'top_k': top_k,
                    'results': results
                }, f, indent=2, ensure_ascii=False)
            print(f"Sonuçlar kaydedildi: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"Arama hatası: {str(e)}")
        return False


def main():
    """Ana program giriş noktası."""
    parser = argparse.ArgumentParser(description='Lost Baggage System (LBS)')
    subparsers = parser.add_subparsers(dest='command', help='Kullanılacak komut')
    
    # Analyze komutu
    analyze_parser = subparsers.add_parser('analyze', help='Bagaj analizi')
    analyze_parser.add_argument('--image', required=True, help='Analiz edilecek resim dosyası')
    analyze_parser.add_argument('--output', help='Çıktı JSON dosyası')
    analyze_parser.add_argument('--config', help='Konfigürasyon dosyası')
    analyze_parser.add_argument('--no-masking', action='store_true', help='Maskeleme kullanma')
    analyze_parser.add_argument('--force-masking', action='store_true', help='Maskeleme zorla kullan')
    
    # Batch komutu
    batch_parser = subparsers.add_parser('batch', help='Toplu analiz')
    batch_parser.add_argument('--input', required=True, help='Girdi klasörü')
    batch_parser.add_argument('--output', required=True, help='Çıktı klasörü')
    batch_parser.add_argument('--config', help='Konfigürasyon dosyası')
    batch_parser.add_argument('--no-masking', action='store_true', help='Maskeleme kullanma')
    batch_parser.add_argument('--force-masking', action='store_true', help='Maskeleme zorla kullan')
    
    # Search komutu
    search_parser = subparsers.add_parser('search', help='Bagaj arama')
    search_parser.add_argument('--query', required=True, help='Sorgu JSON dosyası')
    search_parser.add_argument('--gallery', required=True, help='Galeri klasörü')
    search_parser.add_argument('--output', help='Sonuç dosyası')
    search_parser.add_argument('--top-k', type=int, default=10, help='En iyi K sonuç')
    search_parser.add_argument('--mode', choices=['global', 'group_based', 'multi_image'],
                              default='global', help='Arama modu')
    search_parser.add_argument('--config', help='Konfigürasyon dosyası')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Konfigürasyon yükle
    config = load_config(args.config if hasattr(args, 'config') else None)
    
    # Komutları çalıştır
    if args.command == 'analyze':
        # Maskeleme tercihini belirle
        use_masking = None
        if args.no_masking:
            use_masking = False
        elif args.force_masking:
            use_masking = True
            
        success = analyze_single_image(args.image, args.output, config, use_masking)
    elif args.command == 'batch':
        # Maskeleme tercihini belirle
        use_masking = None
        if args.no_masking:
            use_masking = False
        elif args.force_masking:
            use_masking = True
            
        success = analyze_batch(args.input, args.output, config, use_masking)
    elif args.command == 'search':
        success = search_luggage(args.query, args.gallery, args.output, 
                               config, args.top_k, args.mode)
    else:
        print(f"Bilinmeyen komut: {args.command}")
        success = False
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
