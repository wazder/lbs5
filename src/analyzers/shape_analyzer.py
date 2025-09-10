"""
ShapeAnalyzer - Şekil ve boyut analizi modülü
Croplanmış bagaj fotoğrafları için şekil analizi yapar.
Tüm resim bagaj olarak kabul edilir.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Any


class ShapeAnalyzer:
    """
    Bagaj şekillerini ve boyutlarını analiz eden sınıf.
    Croplanmış fotoğraflar için optimize edilmiş.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        ShapeAnalyzer constructor
        
        Args:
            config: Şekil analizi konfigürasyonu
        """
        self.config = config or {
            "wheel_detection_enabled": True,
            "handle_detection_enabled": True
        }
        
        # Bagaj tipleri için şekil özellikleri
        self.suitcase_templates = {
            "hardshell": {"aspect_ratio": (0.6, 1.4), "rectangularity": 0.7},
            "softshell": {"aspect_ratio": (0.5, 1.5), "rectangularity": 0.6},
            "duffel": {"aspect_ratio": (1.5, 3.0), "rectangularity": 0.4},
            "backpack": {"aspect_ratio": (0.4, 0.8), "rectangularity": 0.5}
        }
    
    def analyze(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Şekil ve boyut analizi yapar.
        Tüm resim bagaj olarak kabul edilir (croplanmış fotoğraf).
        
        Args:
            image: Analiz edilecek resim (croplanmış bagaj fotoğrafı)
            
        Returns:
            Şekil ve boyut bilgilerini içeren sözlük
        """
        try:
            # Resmin kendisi bagaj (crop edilmiş)
            img_height, img_width = image.shape[:2]
            
            # Bagaj tipi tespiti (resim aspect ratio'su üzerinden)
            suitcase_type = self._classify_suitcase_type_from_image(image)
            
            # Boyut analizi
            size_analysis = self._analyze_size_from_image(image)
            
            # Tekerlek tespiti
            wheel_info = self._detect_wheels(image)
            
            # Sap analizi
            handle_info = self._detect_handles(image)
            
            return {
                "suitcase_type": suitcase_type["type"],
                "type_confidence": suitcase_type["confidence"],
                "size_class": size_analysis["class"],
                "dimensions": size_analysis["dimensions"],
                "volume_estimate": size_analysis["volume"],
                "wheel_count": wheel_info["count"],
                "wheel_positions": wheel_info["positions"],
                "handle_types": handle_info["types"],
                "handle_positions": handle_info["positions"],
                "aspect_ratio": size_analysis["aspect_ratio"],
                "rectangularity": size_analysis["rectangularity"]
            }
            
        except Exception as e:
            return {"error": f"Şekil analizi hatası: {str(e)}"}
    
    def _classify_suitcase_type_from_image(self, image: np.ndarray) -> Dict[str, Any]:
        """Resmin genel özelliklerinden bagaj tipini sınıflandırır."""
        h, w = image.shape[:2]
        aspect_ratio = w / h
        
        # Kenar tespiti ile rectangularity hesapla
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Genel şeklin ne kadar dikdörtgensel olduğunu hesapla
        # Kenar piksel sayısı / toplam piksel sayısı
        edge_density = np.sum(edges > 0) / (h * w)
        
        # Basit rectangularity tahmini
        rectangularity = 0.8 if edge_density > 0.1 else 0.5
        
        # Her tip için skor hesapla
        type_scores = {}
        
        for suitcase_type, template in self.suitcase_templates.items():
            score = 0.0
            
            # Aspect ratio skoru
            ar_min, ar_max = template["aspect_ratio"]
            if ar_min <= aspect_ratio <= ar_max:
                ar_score = 1.0 - abs(aspect_ratio - (ar_min + ar_max) / 2) / ((ar_max - ar_min) / 2)
                score += ar_score * 0.7
            
            # Rectangularity skoru
            rect_target = template["rectangularity"]
            rect_score = 1.0 - abs(rectangularity - rect_target)
            score += max(0, rect_score) * 0.3
            
            type_scores[suitcase_type] = score
        
        # En yüksek skorlu tip
        best_type = max(type_scores, key=type_scores.get)
        confidence = type_scores[best_type]
        
        return {
            "type": best_type if confidence > 0.3 else "unknown",
            "confidence": confidence,
            "all_scores": type_scores,
            "metrics": {
                "aspect_ratio": aspect_ratio,
                "rectangularity": rectangularity
            }
        }
    
    def _analyze_size_from_image(self, image: np.ndarray) -> Dict[str, Any]:
        """Resim boyutlarından bagaj boyutunu analiz eder."""
        img_height, img_width = image.shape[:2]
        
        # Aspect ratio
        aspect_ratio = img_width / img_height
        
        # Piksel sayısına göre boyut sınıfı (croplanmış resim olduğu için)
        total_pixels = img_width * img_height
        
        if total_pixels < 100000:  # 316x316'dan küçük
            size_class = "small"
        elif total_pixels < 400000:  # 632x632'dan küçük
            size_class = "medium"  
        elif total_pixels < 900000:  # 948x948'dan küçük
            size_class = "large"
        else:
            size_class = "oversized"
        
        # Hacim tahmini (basit)
        estimated_depth = min(img_width, img_height) * 0.4
        volume_pixels = img_width * img_height * estimated_depth
        
        # Yaklaşık litre dönüşümü
        pixel_to_cm = 0.25
        volume_cm3 = volume_pixels * (pixel_to_cm ** 3)
        volume_liters = volume_cm3 / 1000
        
        # Rectangularity hesapla (basit yaklaşım)
        rectangularity = 0.8  # Croplanmış resim olduğu için varsayılan
        
        return {
            "class": size_class,
            "dimensions": {
                "width_pixels": img_width,
                "height_pixels": img_height,
                "relative_width": 1.0,  # Croplanmış resim
                "relative_height": 1.0
            },
            "volume": volume_liters,
            "aspect_ratio": aspect_ratio,
            "rectangularity": rectangularity
        }
    
    def _detect_wheels(self, image: np.ndarray) -> Dict[str, Any]:
        """Tekerlek tespiti yapar."""
        if not self.config["wheel_detection_enabled"]:
            return {"count": 0, "positions": []}
        
        # Gri tonlama
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Hough Circle Transform ile tekerlek tespiti
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=30,
            param1=50,
            param2=30,
            minRadius=5,
            maxRadius=min(gray.shape) // 10  # Resim boyutuna göre maksimum radius
        )
        
        detected_wheels = []
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            
            for (x, y, r) in circles:
                detected_wheels.append({
                    "center": (int(x), int(y)),
                    "radius": int(r)
                })
        
        return {
            "count": len(detected_wheels),
            "positions": detected_wheels
        }
    
    def _detect_handles(self, image: np.ndarray) -> Dict[str, Any]:
        """Sap tespiti yapar."""
        if not self.config["handle_detection_enabled"]:
            return {"types": [], "positions": []}
        
        # Basit sap tespiti - uzun ince yapılar ara
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Çizgi tespiti
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=50,
            minLineLength=30,
            maxLineGap=10
        )
        
        handle_types = []
        handle_positions = []
        
        if lines is not None:
            img_height, img_width = gray.shape
            
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                # Çizgi uzunluğu ve açısı
                length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                angle = np.arctan2(y2-y1, x2-x1) * 180 / np.pi
                
                # Teleskopik sap (dikey çizgiler)
                if 80 <= abs(angle) <= 100 and length > img_height * 0.2:
                    handle_types.append("telescopic")
                    handle_positions.append({"type": "telescopic", "line": (x1, y1, x2, y2)})
                
                # Yan taşıma sapı (yatay çizgiler)
                elif abs(angle) <= 20 and length > img_width * 0.2:
                    handle_types.append("side_carry")
                    handle_positions.append({"type": "side_carry", "line": (x1, y1, x2, y2)})
        
        # Üst taşıma sapı tespiti (resmin üst kısmındaki çıkıntılar)
        top_region = gray[:gray.shape[0]//4, :]  # Üst %25'lik bölge
        
        # Üst bölgede küçük konturlar ara
        edges_top = cv2.Canny(top_region, 30, 100)
        contours_top, _ = cv2.findContours(edges_top, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours_top:
            area = cv2.contourArea(contour)
            if 100 < area < 1000:  # Küçük ama anlamlı alanlar
                handle_types.append("top_carry")
                # Konturun merkezi
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    handle_positions.append({"type": "top_carry", "center": (cx, cy)})
        
        return {
            "types": list(set(handle_types)),
            "positions": handle_positions
        }
