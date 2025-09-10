import cv2
import numpy as np
from typing import Optional, List, Tuple, Dict, Any
from pathlib import Path
import logging
import os
import shutil
from PIL import Image

from .sam_detector import SAMDetector

class MaskProcessor:
    """
    Maskeleme işlemlerini yöneten ana sınıf
    """
    
    def __init__(self, 
                 input_dir: str = "data/input",
                 output_dir: str = "data/input-masked",
                 sam_config: Optional[Dict[str, Any]] = None):
        """
        Mask işlemcisini başlat
        
        Args:
            input_dir: Giriş resimlerinin bulunduğu klasör
            output_dir: Maskelenmiş resimlerin kaydedileceği klasör
            sam_config: SAM konfigürasyonu
        """
        self.logger = logging.getLogger(__name__)
        
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        
        # Çıktı klasörünü oluştur
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # SAM dedektörünü başlat
        if sam_config is None:
            sam_config = {
                "model_type": "vit_h",
                "device": "auto"
            }
        
        self.sam_detector = SAMDetector(**sam_config)
        
        # Desteklenen formatlar
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.heic', '.heif'}
        
        self.logger.info(f"MaskProcessor başlatıldı: {input_dir} -> {output_dir}")
    
    def process_all_images(self, 
                          overwrite: bool = False,
                          save_masks: bool = True,
                          quality: int = 95) -> Dict[str, Any]:
        """
        Tüm giriş resimlerini işle
        
        Args:
            overwrite: Var olan dosyaları üzerine yaz
            save_masks: Maskeleri ayrı dosyalar olarak kaydet
            quality: JPEG kalitesi (0-100)
            
        Returns:
            İşlem sonuçları
        """
        results = {
            "processed": [],
            "skipped": [],
            "errors": [],
            "total_files": 0,
            "success_count": 0
        }
        
        if not self.input_dir.exists():
            self.logger.error(f"Giriş klasörü bulunamadı: {self.input_dir}")
            return results
        
        # Tüm desteklenen resim dosyalarını bul
        image_files = []
        for ext in self.supported_formats:
            image_files.extend(self.input_dir.glob(f"*{ext}"))
            image_files.extend(self.input_dir.glob(f"*{ext.upper()}"))
        
        results["total_files"] = len(image_files)
        
        if not image_files:
            self.logger.warning(f"Giriş klasöründe resim dosyası bulunamadı: {self.input_dir}")
            return results
        
        self.logger.info(f"{len(image_files)} resim dosyası bulundu")
        
        for image_file in image_files:
            try:
                result = self.process_single_image(
                    image_file, 
                    overwrite=overwrite,
                    save_mask=save_masks,
                    quality=quality
                )
                
                if result["success"]:
                    results["processed"].append(result)
                    results["success_count"] += 1
                    self.logger.info(f"İşlendi: {image_file.name}")
                else:
                    results["errors"].append({
                        "file": str(image_file),
                        "error": result.get("error", "Bilinmeyen hata")
                    })
                    self.logger.error(f"Hata: {image_file.name} - {result.get('error')}")
                    
            except Exception as e:
                error_msg = f"İşlenirken hata: {e}"
                results["errors"].append({
                    "file": str(image_file),
                    "error": error_msg
                })
                self.logger.error(f"Kritik hata: {image_file.name} - {error_msg}")
        
        self.logger.info(f"İşlem tamamlandı: {results['success_count']}/{results['total_files']} başarılı")
        return results
    
    def process_single_image(self, 
                           image_path: Path,
                           overwrite: bool = False,
                           save_mask: bool = True,
                           quality: int = 95) -> Dict[str, Any]:
        """
        Tek bir resmi işle
        
        Args:
            image_path: İşlenecek resim dosyası
            overwrite: Var olan dosyayı üzerine yaz
            save_mask: Maskeyi ayrı dosya olarak kaydet
            quality: JPEG kalitesi
            
        Returns:
            İşlem sonucu
        """
        result = {
            "success": False,
            "input_file": str(image_path),
            "output_file": None,
            "mask_file": None,
            "error": None
        }
        
        try:
            # Çıktı dosya yollarını oluştur
            output_path = self.output_dir / f"{image_path.stem}_masked{image_path.suffix}"
            mask_path = self.output_dir / f"{image_path.stem}_mask.png"
            
            # Dosya varsa ve overwrite False ise atla
            if output_path.exists() and not overwrite:
                result["error"] = "Dosya zaten var ve overwrite=False"
                return result
            
            # Resmi yükle
            image = self._load_image(image_path)
            if image is None:
                result["error"] = "Resim yüklenemedi"
                return result
            
            # Bagaj tespiti ve maske oluştur
            mask = self.sam_detector.detect_luggage(image)
            if mask is None:
                result["error"] = "Bagaj tespiti başarısız"
                return result
            
            # Maskelenmiş resim oluştur
            masked_image = self._apply_mask(image, mask)
            
            # Çıktıları kaydet
            success_save = self._save_image(masked_image, output_path, quality)
            if not success_save:
                result["error"] = "Maskelenmiş resim kaydedilemedi"
                return result
            
            result["output_file"] = str(output_path)
            
            # Maskeyi kaydet
            if save_mask:
                mask_success = self._save_mask(mask, mask_path)
                if mask_success:
                    result["mask_file"] = str(mask_path)
            
            result["success"] = True
            
        except Exception as e:
            result["error"] = str(e)
        
        return result
    
    def _load_image(self, image_path: Path) -> Optional[np.ndarray]:
        """
        Resim dosyasını yükle
        """
        try:
            if image_path.suffix.lower() in ['.heic', '.heif']:
                # HEIC/HEIF için özel yükleme
                from pillow_heif import register_heif_opener
                register_heif_opener()
                
                pil_image = Image.open(image_path)
                if pil_image.mode == 'RGBA':
                    pil_image = pil_image.convert('RGB')
                
                # PIL'den OpenCV formatına çevir
                image_array = np.array(pil_image)
                return cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            else:
                # Standart formatlar için OpenCV kullan
                image = cv2.imread(str(image_path))
                return image
                
        except Exception as e:
            self.logger.error(f"Resim yükleme hatası {image_path}: {e}")
            return None
    
    def _apply_mask(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Maskeyi resme uygula
        """
        # Maskeyi 3 kanallı yap
        if len(mask.shape) == 2:
            mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        else:
            mask_3ch = mask
        
        # Maskeyi normalize et (0-1 arası)
        mask_norm = mask_3ch.astype(np.float32) / 255.0
        
        # Maskeyi uygula
        masked_image = image.astype(np.float32) * mask_norm
        
        # Arka planı beyaz yap
        background = np.ones_like(image, dtype=np.float32) * 255
        inverse_mask = 1.0 - mask_norm
        
        result = masked_image + (background * inverse_mask)
        
        return result.astype(np.uint8)
    
    def _save_image(self, image: np.ndarray, output_path: Path, quality: int = 95) -> bool:
        """
        Resmi kaydet
        """
        try:
            if output_path.suffix.lower() in ['.jpg', '.jpeg']:
                cv2.imwrite(str(output_path), image, [cv2.IMWRITE_JPEG_QUALITY, quality])
            else:
                cv2.imwrite(str(output_path), image)
            return True
        except Exception as e:
            self.logger.error(f"Resim kaydetme hatası {output_path}: {e}")
            return False
    
    def _save_mask(self, mask: np.ndarray, mask_path: Path) -> bool:
        """
        Maskeyi kaydet
        """
        try:
            cv2.imwrite(str(mask_path), mask)
            return True
        except Exception as e:
            self.logger.error(f"Maske kaydetme hatası {mask_path}: {e}")
            return False
    
    def clear_output_directory(self) -> bool:
        """
        Çıktı klasörünü temizle
        """
        try:
            if self.output_dir.exists():
                shutil.rmtree(self.output_dir)
                self.output_dir.mkdir(parents=True, exist_ok=True)
                self.logger.info("Çıktı klasörü temizlendi")
                return True
        except Exception as e:
            self.logger.error(f"Klasör temizleme hatası: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        İşlem istatistiklerini al
        """
        stats = {
            "input_files": 0,
            "output_files": 0,
            "mask_files": 0,
            "input_dir_exists": self.input_dir.exists(),
            "output_dir_exists": self.output_dir.exists()
        }
        
        if self.input_dir.exists():
            input_files = []
            for ext in self.supported_formats:
                input_files.extend(self.input_dir.glob(f"*{ext}"))
                input_files.extend(self.input_dir.glob(f"*{ext.upper()}"))
            stats["input_files"] = len(input_files)
        
        if self.output_dir.exists():
            output_files = list(self.output_dir.glob("*_masked.*"))
            mask_files = list(self.output_dir.glob("*_mask.png"))
            stats["output_files"] = len(output_files)
            stats["mask_files"] = len(mask_files)
        
        return stats
