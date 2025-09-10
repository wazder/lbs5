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
                      confidence_threshold: float = 0.3) -> Optional[np.ndarray]:
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
            
            h, w = image.shape[:2]
            
            # Merkez + köşe noktaları (valizin farklı bölgeleri)
            points = np.array([
                [w//2, h//2],      # Merkez
                [w//3, h//3],      # Sol üst
                [2*w//3, h//3],    # Sağ üst  
                [w//3, 2*h//3],    # Sol alt
                [2*w//3, 2*h//3],  # Sağ alt
            ])
            
            # Tüm noktalar pozitif (valiz içi)
            labels = np.array([1, 1, 1, 1, 1])
            
            # Maskeyi tahmin et (çoklu nokta)
            masks, scores, _ = self.predictor.predict(
                point_coords=points,
                point_labels=labels,
                multimask_output=True
            )
            
            # En iyi maskeyi seç veya kombinasyon kullan
            if len(masks) > 0:
                # Tüm maskeleri birleştir (valiz için daha kapsamlı)
                combined_mask = np.zeros_like(masks[0], dtype=bool)
                for i, (mask, score) in enumerate(zip(masks, scores)):
                    if score >= confidence_threshold:
                        combined_mask = combined_mask | mask
                
                # Eğer hiç mask yoksa en iyisini al
                if not combined_mask.any():
                    best_mask_idx = np.argmax(scores)
                    combined_mask = masks[best_mask_idx]
                    if scores[best_mask_idx] < confidence_threshold:
                        self.logger.warning(f"SAM güven skoru düşük ({scores[best_mask_idx]:.3f}), fallback kullanılıyor")
                        return self._fallback_detection(image)
                
                # Binary mask olarak döndür
                return (combined_mask * 255).astype(np.uint8)
            else:
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
        Bast Tespit
        """
        # Gri tonlamalı resme çevir
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Ön işleme - valiz için optimize
        # 1. Gürültü azaltma
        denoised = cv2.medianBlur(gray, 5)
        
        # 2. Kontrast artırma (valiz kenarları için)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        
        # 3. Çoklu eşik değer ile kenar tespiti
        edges1 = cv2.Canny(enhanced, 30, 100)  # Düşük eşik
        edges2 = cv2.Canny(enhanced, 50, 150)  # Orta eşik
        edges3 = cv2.Canny(enhanced, 80, 200)  # Yüksek eşik
        
        # Kenarları birleştir
        combined_edges = cv2.bitwise_or(edges1, cv2.bitwise_or(edges2, edges3))
        
        # Morfolojik işlemler - valiz şekli için
        # Dikdörtgen kernel (valizler genelde dikdörtgen)
        kernel_rect = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
        # Önce kapatma sonra açma
        closed = cv2.morphologyEx(combined_edges, cv2.MORPH_CLOSE, kernel_rect)
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_ellipse)
        
        # Konturları bul
        contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Valiz kriterleri ile filtreleme
            valid_contours = []
            h, w = image.shape[:2]
            min_area = (w * h) * 0.05  # Minimum %5 alan
            max_area = (w * h) * 0.9   # Maksimum %90 alan
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if min_area < area < max_area:
                    # Aspect ratio kontrolü (valizler genelde 1:1 - 2:1 arası)
                    x, y, cw, ch = cv2.boundingRect(contour)
                    aspect_ratio = max(cw, ch) / min(cw, ch)
                    if aspect_ratio < 3.0:  # Çok uzun değil
                        valid_contours.append((contour, area))
            
            if valid_contours:
                # En büyük geçerli konturu seç
                largest_contour = max(valid_contours, key=lambda x: x[1])[0]
                
                # Mask oluştur
                mask = np.zeros(gray.shape, dtype=np.uint8)
                cv2.fillPoly(mask, [largest_contour], 255)
                
                return mask
        
        # Hiç geçerli kontur bulunamazsa, merkezi bölgeyi valiz olarak kabul et
        h, w = gray.shape
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Merkezi %60'lık alanı valiz olarak işaretle
        center_h, center_w = h // 2, w // 2
        margin_h, margin_w = int(h * 0.3), int(w * 0.3)
        
        mask[margin_h:h-margin_h, margin_w:w-margin_w] = 255
        
        return mask
