"""
ImageProcessor - Resim ön işleme ve yardımcı fonksiyonlar
"""

import cv2
import numpy as np
from typing import Optional, Tuple
import os
from PIL import Image
import pillow_heif


class ImageProcessor:
    """
    Resim yükleme, ön işleme ve format dönüşümleri için yardımcı sınıf.
    HEIC/HEIF formatı dahil olmak üzere çeşitli formatları destekler.
    """
    
    def __init__(self):
        """ImageProcessor constructor"""
        # HEIC desteğini etkinleştir
        pillow_heif.register_heif_opener()
        
        self.supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.heic', '.heif']
    
    def load_and_preprocess(self, image_path: str, 
                          target_size: Optional[Tuple[int, int]] = None) -> Optional[np.ndarray]:
        """
        Resmi yükler ve ön işleme yapar.
        
        Args:
            image_path: Resim dosyası yolu
            target_size: Hedef boyut (width, height)
            
        Returns:
            Ön işlenmiş resim veya None
        """
        try:
            # Dosya var mı kontrol et
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Resim dosyası bulunamadı: {image_path}")
            
            # Format kontrolü
            file_ext = os.path.splitext(image_path)[1].lower()
            if file_ext not in self.supported_formats:
                raise ValueError(f"Desteklenmeyen format: {file_ext}")
            
            # HEIC/HEIF dosyalar için özel yükleme
            if file_ext in ['.heic', '.heif']:
                image = self._load_heic_image(image_path)
            else:
                # OpenCV ile standart formatları yükle
                image = cv2.imread(image_path)
            
            if image is None:
                raise ValueError(f"Resim yüklenemedi: {image_path}")
            
            # Boyut ayarlama
            if target_size:
                image = cv2.resize(image, target_size)
            
            # Temel ön işleme
            image = self._basic_preprocessing(image)
            
            return image
            
        except Exception as e:
            print(f"Resim işleme hatası: {str(e)}")
            return None
    
    def _load_heic_image(self, image_path: str) -> Optional[np.ndarray]:
        """HEIC/HEIF formatındaki resimleri yükler."""
        try:
            # PIL ile HEIC dosyayı aç
            pil_image = Image.open(image_path)
            
            # RGB'ye çevir (RGBA olabilir)
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # NumPy array'e çevir
            image_array = np.array(pil_image)
            
            # PIL RGB -> OpenCV BGR
            image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            
            return image_bgr
            
        except Exception as e:
            print(f"HEIC yükleme hatası: {str(e)}")
            return None
    
    def _basic_preprocessing(self, image: np.ndarray) -> np.ndarray:
        """Temel ön işleme adımları."""
        # Gürültü azaltma
        denoised = cv2.bilateralFilter(image, 9, 75, 75)
        
        # Kontrastı iyileştir
        lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    def resize_maintain_aspect(self, image: np.ndarray, 
                             max_dimension: int = 800) -> np.ndarray:
        """Aspect ratio koruyarak yeniden boyutlandır."""
        h, w = image.shape[:2]
        
        if max(h, w) <= max_dimension:
            return image
        
        if h > w:
            new_h = max_dimension
            new_w = int(w * max_dimension / h)
        else:
            new_w = max_dimension
            new_h = int(h * max_dimension / w)
        
        return cv2.resize(image, (new_w, new_h))
    
    def extract_roi(self, image: np.ndarray, 
                   bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """Belirtilen bölgeyi çıkarır."""
        x, y, w, h = bbox
        return image[y:y+h, x:x+w]
    
    def save_image(self, image: np.ndarray, output_path: str) -> bool:
        """Resmi kaydet."""
        try:
            return cv2.imwrite(output_path, image)
        except Exception:
            return False
