import cv2
import numpy as np
import torch
from typing import Optional, List, Tuple, Dict, Any
from pathlib import Path
import logging

try:
    from segment_anything import sam_model_registry, SamPredictor
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False
    
try:
    from rembg import remove, new_session
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False

class SAMDetector:
    """
    SAM (Segment Anything Model) tabanlı bagaj tespit sistemi
    """
    
    def __init__(self, 
                 model_type: str = "vit_h",
                 checkpoint_path: Optional[str] = None,
                 device: str = "auto"):
        """
        SAM dedektörünü başlat
        
        Args:
            model_type: SAM model tipi ('vit_h', 'vit_l', 'vit_b')
            checkpoint_path: Model ağırlık dosyası yolu
            device: Hesaplama cihazı ('cpu', 'cuda', 'auto')
        """
        self.logger = logging.getLogger(__name__)
        
        if not SAM_AVAILABLE:
            self.logger.warning("SAM kütüphanesi yüklü değil. Alternatif yöntemler kullanılacak.")
            self.sam_available = False
            return
            
        self.sam_available = True
        self.model_type = model_type
        
        # Cihaz seçimi
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.logger.info(f"SAM cihazı: {self.device}")
        
        # Model yükleme
        try:
            if checkpoint_path and Path(checkpoint_path).exists():
                self.sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
            else:
                self.logger.warning("SAM checkpoint bulunamadı. Model indirilecek...")
                self.sam = sam_model_registry[model_type]()
                
            self.sam.to(device=self.device)
            self.predictor = SamPredictor(self.sam)
            self.logger.info(f"SAM model başarıyla yüklendi: {model_type}")
            
        except Exception as e:
            self.logger.error(f"SAM model yüklenemedi: {e}")
            self.sam_available = False
    
    def detect_luggage(self, 
                      image: np.ndarray,
                      confidence_threshold: float = 0.5) -> Optional[np.ndarray]:
        """
        Resimde bagaj tespiti yap
        
        Args:
            image: Giriş resmi (BGR formatında)
            confidence_threshold: Güven eşiği
            
        Returns:
            Binary mask (bagaj=255, arka plan=0) veya None
        """
        if not self.sam_available:
            return self._fallback_detection(image)
            
        try:
            # RGB formatına çevir
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # SAM için resmi ayarla
            self.predictor.set_image(rgb_image)
            
            # Resmin merkezi noktasını bagaj olarak işaretle
            h, w = image.shape[:2]
            center_point = np.array([[w//2, h//2]])
            center_label = np.array([1])  # Pozitif nokta
            
            # Maskeyi tahmin et
            masks, scores, _ = self.predictor.predict(
                point_coords=center_point,
                point_labels=center_label,
                multimask_output=True
            )
            
            # En iyi maskeyi seç
            best_mask_idx = np.argmax(scores)
            best_mask = masks[best_mask_idx]
            
            if scores[best_mask_idx] >= confidence_threshold:
                # Binary mask olarak döndür
                return (best_mask * 255).astype(np.uint8)
            else:
                self.logger.warning("SAM güven skoru düşük, fallback kullanılıyor")
                return self._fallback_detection(image)
                
        except Exception as e:
            self.logger.error(f"SAM tespiti başarısız: {e}")
            return self._fallback_detection(image)
    
    def _fallback_detection(self, image: np.ndarray) -> np.ndarray:
        """
        SAM olmadığında alternatif tespit yöntemi
        """
        if REMBG_AVAILABLE:
            return self._rembg_detection(image)
        else:
            return self._simple_detection(image)
    
    def _rembg_detection(self, image: np.ndarray) -> np.ndarray:
        """
        RemBG kullanarak arka plan kaldırma
        """
        try:
            # RemBG session oluştur
            session = new_session('u2net')
            
            # PIL formatına çevir
            from PIL import Image
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            # Arka planı kaldır
            output = remove(pil_image, session=session)
            
            # Alpha kanalından mask oluştur
            output_array = np.array(output)
            if output_array.shape[2] == 4:  # RGBA
                mask = output_array[:, :, 3]  # Alpha kanalı
                return mask
            else:
                return self._simple_detection(image)
                
        except Exception as e:
            self.logger.error(f"RemBG tespiti başarısız: {e}")
            return self._simple_detection(image)
    
    def _simple_detection(self, image: np.ndarray) -> np.ndarray:
        """
        Basit kenar tabanlı tespit
        """
        # Gri tonlamalı resme çevir
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Gürültüyü azalt
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Kenar tespiti
        edges = cv2.Canny(blurred, 50, 150)
        
        # Morfolojik işlemler
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # Konturları bul
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # En büyük konturu seç
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Mask oluştur
            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.fillPoly(mask, [largest_contour], 255)
            
            return mask
        else:
            # Hiç kontur bulunamazsa tüm resmi bagaj olarak kabul et
            return np.full(gray.shape, 255, dtype=np.uint8)
