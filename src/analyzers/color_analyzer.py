"""
ColorAnalyzer - Renk analizi modülü
K-means kümeleme ile dominant renk çıkarımı yapar.
"""

import cv2
import numpy as np
from sklearn.cluster import KMeans
from typing import Dict, List, Tuple, Any
import colorsys


class ColorAnalyzer:
    """
    Bagaj renklerini analiz eden sınıf.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        ColorAnalyzer constructor
        
        Args:
            config: Renk analizi konfigürasyonu
        """
        self.config = config or {
            "num_clusters": 3,
            "color_spaces": ["RGB", "HSV", "LAB"]
        }
    
    def analyze(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Renk analizi yapar.
        
        Args:
            image: Analiz edilecek resim (BGR formatında)
            
        Returns:
            Renk bilgilerini içeren sözlük
        """
        try:
            # Dominant renkleri çıkar
            dominant_colors = self._extract_dominant_colors(image)
            
            # Çoklu renk uzayı analizi
            color_spaces_data = self._analyze_color_spaces(image)
            
            # Yüzey analizi (parlaklık vs)
            surface_analysis = self._analyze_surface_properties(image)
            
            return {
                "dominant_colors": dominant_colors,
                "color_spaces": color_spaces_data,
                "finish": surface_analysis["finish"],
                "surface_properties": surface_analysis
            }
            
        except Exception as e:
            return {"error": f"Renk analizi hatası: {str(e)}"}
    
    def _extract_dominant_colors(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """K-means ile dominant renkleri çıkarır."""
        # Resmi düzleştir
        pixels = image.reshape(-1, 3)
        
        # K-means kümeleme
        kmeans = KMeans(
            n_clusters=self.config["num_clusters"], 
            random_state=42,
            n_init=10
        )
        kmeans.fit(pixels)
        
        # Küme merkezleri (dominant renkler)
        colors = kmeans.cluster_centers_.astype(int)
        
        # Her rengin yüzdesini hesapla
        labels = kmeans.labels_
        percentages = np.bincount(labels) / len(labels)
        
        dominant_colors = []
        for i, (color, percentage) in enumerate(zip(colors, percentages)):
            color_info = {
                "rank": i + 1,
                "bgr": color.tolist(),
                "rgb": [color[2], color[1], color[0]],  # BGR'den RGB'ye
                "percentage": float(percentage),
                "hex": self._bgr_to_hex(color),
                "name": self._get_color_name(color)
            }
            dominant_colors.append(color_info)
        
        # Yüzdeye göre sırala
        dominant_colors.sort(key=lambda x: x["percentage"], reverse=True)
        
        return dominant_colors
    
    def _analyze_color_spaces(self, image: np.ndarray) -> Dict[str, Any]:
        """Farklı renk uzaylarında analiz yapar."""
        color_spaces = {}
        
        # RGB analizi
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        color_spaces["rgb"] = {
            "mean": np.mean(rgb_image, axis=(0, 1)).tolist(),
            "std": np.std(rgb_image, axis=(0, 1)).tolist()
        }
        
        # HSV analizi
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        color_spaces["hsv"] = {
            "mean": np.mean(hsv_image, axis=(0, 1)).tolist(),
            "std": np.std(hsv_image, axis=(0, 1)).tolist()
        }
        
        # LAB analizi
        lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        color_spaces["lab"] = {
            "mean": np.mean(lab_image, axis=(0, 1)).tolist(),
            "std": np.std(lab_image, axis=(0, 1)).tolist()
        }
        
        return color_spaces
    
    def _analyze_surface_properties(self, image: np.ndarray) -> Dict[str, Any]:
        """Yüzey özelliklerini analiz eder."""
        # Gri tonlamaya çevir
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Gradyan hesapla (yüzey pürüzlülüğü için)
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Yüzey özelliklerini hesapla
        surface_roughness = np.std(gradient_magnitude)
        brightness_variance = np.std(gray)
        
        # Yüzey tipi belirle
        if surface_roughness > 50:
            finish = "textured"
        elif brightness_variance > 40:
            finish = "glossy"
        else:
            finish = "matte"
        
        return {
            "finish": finish,
            "roughness": float(surface_roughness),
            "brightness_variance": float(brightness_variance),
            "reflectivity": "high" if brightness_variance > 40 else "low"
        }
    
    def _bgr_to_hex(self, bgr_color: np.ndarray) -> str:
        """BGR rengini hex koduna çevirir."""
        return "#{:02x}{:02x}{:02x}".format(
            int(bgr_color[2]), int(bgr_color[1]), int(bgr_color[0])
        )
    
    def _get_color_name(self, bgr_color: np.ndarray) -> str:
        """BGR rengine isim atar."""
        # Basit renk isimlendirme
        r, g, b = bgr_color[2], bgr_color[1], bgr_color[0]
        
        if r > 200 and g > 200 and b > 200:
            return "white"
        elif r < 50 and g < 50 and b < 50:
            return "black"
        elif r > g and r > b:
            return "red"
        elif g > r and g > b:
            return "green"
        elif b > r and b > g:
            return "blue"
        elif r > 150 and g > 150 and b < 100:
            return "yellow"
        elif r > 150 and g < 100 and b > 150:
            return "purple"
        elif r < 100 and g > 150 and b > 150:
            return "cyan"
        else:
            return "mixed"
