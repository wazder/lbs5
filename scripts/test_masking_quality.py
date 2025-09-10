#!/usr/bin/env python3
"""
LBS5 Maskeleme Test ve Optimizasyon Scripti
Valiz tespiti kalitesini test eder ve farklı parametrelerle karşılaştırır
"""

import sys
import os
from pathlib import Path
import json
import time
import numpy as np
import cv2

# Proje kök dizinini sys.path'e ekle
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.mask.sam_detector import SAMDetector
from src.mask.mask_processor import MaskProcessor

def test_masking_quality(image_path: str, output_dir: str = "test_results"):
    """
    Maskeleme kalitesini farklı parametrelerle test eder
    """
    print(f"VALİZ MASKELEME KALİTE TESTİ")
    print(f"Test resmi: {image_path}")
    print(50)

    # Çıktı klasörünü oluştur
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Orijinal resmi yükle
    original = cv2.imread(image_path)
    if original is None:
        print(f"Resim yüklenemedi: {image_path}")
        return
    
    h, w = original.shape[:2]
    print(f"📐 Resim boyutu: {w}x{h}")
    
    # Test konfigürasyonları
    test_configs = [
        {
            "name": "Yüksek_Eşik_Tek_Nokta",
            "confidence_threshold": 0.7,
            "multi_point": False
        },
        {
            "name": "Orta_Eşik_Tek_Nokta", 
            "confidence_threshold": 0.5,
            "multi_point": False
        },
        {
            "name": "Düşük_Eşik_Çoklu_Nokta",
            "confidence_threshold": 0.3,
            "multi_point": True
        },
        {
            "name": "Çok_Düşük_Eşik_Çoklu",
            "confidence_threshold": 0.1,
            "multi_point": True
        }
    ]
    
    results = []
    
    for config in test_configs:
        print(f"\n🔬 Test: {config['name']}")
        print(f"   Eşik: {config['confidence_threshold']}")
        print(f"   Çoklu nokta: {config['multi_point']}")
        
        try:
            # SAM Detector oluştur
            detector = SAMDetector(
                model_type="vit_h",
                device="cuda" if "cuda" in str(original) else "auto"
            )
            
            start_time = time.time()
            
            # Maskeyi al
            mask = detector.detect_luggage(
                original, 
                confidence_threshold=config['confidence_threshold']
            )
            
            process_time = time.time() - start_time
            
            if mask is not None:
                # Mask kalitesini değerlendir
                quality_metrics = evaluate_mask_quality(original, mask)
                
                # Maskelenmiş resmi oluştur
                masked_image = apply_mask_with_white_bg(original, mask)
                
                # Sonuçları kaydet
                config_name = config['name']
                cv2.imwrite(str(output_path / f"{config_name}_mask.png"), mask)
                cv2.imwrite(str(output_path / f"{config_name}_result.jpg"), masked_image)
                
                # Sonuçları topla
                result = {
                    "config": config,
                    "process_time": process_time,
                    "quality": quality_metrics,
                    "success": True
                }
                results.append(result)
                
                print(f"   Başarılı - {process_time:.2f}s")
                print(f"   Kaplama: %{quality_metrics['coverage']*100:.1f}")
                print(f"   Kenar kalitesi: {quality_metrics['edge_quality']:.3f}")
                
            else:
                print(f"   Mask oluşturulamadı")
                results.append({
                    "config": config,
                    "process_time": process_time,
                    "success": False
                })
                
        except Exception as e:
            print(f"   Hata: {e}")
            results.append({
                "config": config,
                "error": str(e),
                "success": False
            })
    
    # Sonuçları kaydet
    with open(output_path / "test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # En iyi sonucu bul
    successful_results = [r for r in results if r.get("success", False)]
    if successful_results:
        best_result = max(successful_results, 
                         key=lambda x: x["quality"]["coverage"])
        
        print(f"\nEN İYİ SONUÇ:")
        print(f"   Konfigürasyon: {best_result['config']['name']}")
        print(f"   Kaplama: %{best_result['quality']['coverage']*100:.1f}")
        print(f"   Süre: {best_result['process_time']:.2f}s")
        
        return best_result['config']
    else:
        print(f"\nHiçbir test başarılı olmadı")
        return None

def evaluate_mask_quality(original: np.ndarray, mask: np.ndarray) -> dict:
    """
    Mask kalitesini değerlendir
    """
    h, w = original.shape[:2]
    total_pixels = h * w
    
    # Mask kapsamı
    mask_pixels = np.sum(mask > 0)
    coverage = mask_pixels / total_pixels
    
    # Kenar kalitesi (mask'in kenar yumuşaklığı)
    edges = cv2.Canny(mask, 50, 150)
    edge_pixels = np.sum(edges > 0)
    edge_quality = edge_pixels / mask_pixels if mask_pixels > 0 else 0
    
    # Merkezi konum (valiz genelde merkezde)
    moments = cv2.moments(mask)
    if moments["m00"] != 0:
        center_x = int(moments["m10"] / moments["m00"])
        center_y = int(moments["m01"] / moments["m00"])
        
        image_center_x, image_center_y = w // 2, h // 2
        center_distance = np.sqrt((center_x - image_center_x)**2 + 
                                (center_y - image_center_y)**2)
        center_normalized = 1.0 - (center_distance / (w/2))  # 0-1 arası
    else:
        center_normalized = 0
    
    # Şekil düzgünlüğü (convex hull ile karşılaştırma)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(largest_contour)
        hull_area = cv2.contourArea(hull)
        contour_area = cv2.contourArea(largest_contour)
        shape_regularity = contour_area / hull_area if hull_area > 0 else 0
    else:
        shape_regularity = 0
    
    return {
        "coverage": coverage,
        "edge_quality": edge_quality,
        "center_alignment": center_normalized,
        "shape_regularity": shape_regularity,
        "overall_score": (coverage * 0.4 + 
                         center_normalized * 0.3 + 
                         shape_regularity * 0.2 + 
                         edge_quality * 0.1)
    }

def apply_mask_with_white_bg(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Maskeyi beyaz arka plan ile uygula
    """
    # Maskeyi 3 kanallı yap
    if len(mask.shape) == 2:
        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    else:
        mask_3ch = mask
    
    # Maskeyi normalize et
    mask_norm = mask_3ch.astype(np.float32) / 255.0
    
    # Maskeyi uygula
    masked_image = image.astype(np.float32) * mask_norm
    
    # Beyaz arka plan ekle
    background = np.ones_like(image, dtype=np.float32) * 255
    inverse_mask = 1.0 - mask_norm
    
    result = masked_image + (background * inverse_mask)
    
    return result.astype(np.uint8)

def batch_test_directory(input_dir: str, output_dir: str = "batch_test_results"):
    """
    Bir klasördeki tüm resimleri test et
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Desteklenen formatlar
    formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    
    # Resim dosyalarını bul
    image_files = []
    for fmt in formats:
        image_files.extend(input_path.glob(f"*{fmt}"))
        image_files.extend(input_path.glob(f"*{fmt.upper()}"))
    
    print(f" {len(image_files)} resim dosyası bulundu")
    
    if not image_files:
        print("Hiç resim dosyası bulunamadı")
        return
    
    # Her resim için test
    all_results = {}
    for i, image_file in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}] {image_file.name}")
        
        file_output_dir = output_path / image_file.stem
        best_config = test_masking_quality(str(image_file), str(file_output_dir))
        all_results[image_file.name] = best_config
    
    # Genel sonuçları kaydet
    with open(output_path / "batch_summary.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n📊 TOPLU TEST TAMAMLANDI")
    print(f"📁 Sonuçlar: {output_path}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="LBS5 Maskeleme Kalite Testi")
    parser.add_argument("--image", help="Test edilecek resim")
    parser.add_argument("--batch", help="Test edilecek klasör")
    parser.add_argument("--output", default="test_results", help="Çıktı klasörü")
    
    args = parser.parse_args()
    
    if args.image:
        test_masking_quality(args.image, args.output)
    elif args.batch:
        batch_test_directory(args.batch, args.output)
    else:
        # Varsayılan: data/input klasörünü test et
        batch_test_directory("data/input", args.output)
