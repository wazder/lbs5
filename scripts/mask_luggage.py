#!/usr/bin/env python3
"""
LBS Maskeleme CLI Aracı
Bagaj resimlerinden arka planı kaldırarak maskelenmiş versiyonlar oluşturur.
"""

import argparse
import logging
import sys
from pathlib import Path

# Projenin kök dizinini sys.path'e ekle
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.mask.mask_processor import MaskProcessor

def setup_logging(verbose: bool = False):
    """
    Logging sistemini ayarla
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

def main():
    """
    Ana fonksiyon
    """
    parser = argparse.ArgumentParser(
        description="LBS Bagaj Maskeleme Sistemi",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Örnekler:
  python scripts/mask_luggage.py                    # Varsayılan klasörleri kullan
  python scripts/mask_luggage.py -i photos -o masked # Özel klasörler
  python scripts/mask_luggage.py --clear-output     # Çıktı klasörünü temizle
  python scripts/mask_luggage.py --stats           # İstatistikleri göster
        """
    )
    
    parser.add_argument(
        "-i", "--input-dir",
        default="data/input",
        help="Giriş resimlerinin bulunduğu klasör (varsayılan: data/input)"
    )
    
    parser.add_argument(
        "-o", "--output-dir", 
        default="data/input-masked",
        help="Maskelenmiş resimlerin kaydedileceği klasör (varsayılan: data/input-masked)"
    )
    
    parser.add_argument(
        "--model-type",
        choices=["vit_h", "vit_l", "vit_b"],
        default="vit_h",
        help="SAM model tipi (varsayılan: vit_h)"
    )
    
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Hesaplama cihazı (varsayılan: auto)"
    )
    
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Var olan dosyaları üzerine yaz"
    )
    
    parser.add_argument(
        "--no-save-masks",
        action="store_true",
        help="Maskeleri ayrı dosyalar olarak kaydetme"
    )
    
    parser.add_argument(
        "--quality",
        type=int,
        default=95,
        help="JPEG kalitesi (0-100, varsayılan: 95)"
    )
    
    parser.add_argument(
        "--clear-output",
        action="store_true",
        help="İşlemden önce çıktı klasörünü temizle"
    )
    
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Sadece istatistikleri göster"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Detaylı çıktı"
    )
    
    args = parser.parse_args()
    
    # Logging'i ayarla
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    try:
        # SAM konfigürasyonu
        sam_config = {
            "model_type": args.model_type,
            "device": args.device
        }
        
        # MaskProcessor'ı başlat
        processor = MaskProcessor(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            sam_config=sam_config
        )
        
        # İstatistikleri göster
        if args.stats:
            stats = processor.get_statistics()
            print("\nLBS Maskeleme İstatistikleri:")
            print(f"├── Giriş klasörü: {args.input_dir}")
            print(f"├── Çıktı klasörü: {args.output_dir}")
            print(f"├── Giriş dosyaları: {stats['input_files']}")
            print(f"├── Çıktı dosyaları: {stats['output_files']}")
            print(f"└── Maske dosyaları: {stats['mask_files']}")
            return 0
        
        # Çıktı klasörünü temizle
        if args.clear_output:
            logger.info("Çıktı klasörü temizleniyor...")
            if processor.clear_output_directory():
                logger.info("Çıktı klasörü temizlendi")
            else:
                logger.error("Çıktı klasörü temizlenemedi")
                return 1
        
        # Ana işlemi başlat
        logger.info("Bagaj maskeleme işlemi başlıyor...")
        logger.info(f"Giriş: {args.input_dir}")
        logger.info(f"Çıktı: {args.output_dir}")
        logger.info(f"Model: {args.model_type} ({args.device})")
        
        results = processor.process_all_images(
            overwrite=args.overwrite,
            save_masks=not args.no_save_masks,
            quality=args.quality
        )
        
        # Sonuçları göster
        print(f"\nİşlem Sonuçları:")
        print(f"├── Toplam dosya: {results['total_files']}")
        print(f"├── Başarılı: {results['success_count']}")
        print(f"├── Atlanan: {len(results['skipped'])}")
        print(f"└── Hatalı: {len(results['errors'])}")
        
        if results['errors']:
            print("\nHatalar:")
            for error in results['errors'][:5]:  # İlk 5 hatayı göster
                print(f"   • {Path(error['file']).name}: {error['error']}")
            if len(results['errors']) > 5:
                print(f"   ... ve {len(results['errors']) - 5} hata daha")
        
        if results['success_count'] > 0:
            logger.info("Maskeleme işlemi tamamlandı!")
            return 0
        else:
            logger.error("Hiçbir dosya işlenemedi")
            return 1
            
    except KeyboardInterrupt:
        logger.info("\nİşlem kullanıcı tarafından iptal edildi")
        return 1
    except Exception as e:
        logger.error(f"Kritik hata: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
