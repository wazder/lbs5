"""
TextureAnalyzer - Doku ve yüzey deseni analizi modülü
LBP (Local Binary Patterns) ve diğer doku analizi tekniklerini kullanır.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Any
from scipy import ndimage


class TextureAnalyzer:
    """
    Bagaj dokularını ve yüzey desenlerini analiz eden sınıf.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        TextureAnalyzer constructor
        
        Args:
            config: Doku analizi konfigürasyonu
        """
        self.config = config or {
            "lbp_radius": 3,
            "lbp_n_points": 24,
            "glcm_enabled": True,
            "pattern_detection": True
        }
    
    def analyze(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Doku analizi yapar.
        
        Args:
            image: Analiz edilecek resim
            
        Returns:
            Doku bilgilerini içeren sözlük
        """
        try:
            # Gri tonlamaya çevir
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # LBP analizi
            lbp_features = self._calculate_lbp_features(gray)
            
            # Doku istatistikleri
            texture_stats = self._calculate_texture_statistics(gray)
            
            # Desen tespiti
            pattern_info = self._detect_patterns(gray)
            
            # Pürüzlülük analizi
            roughness = self._analyze_surface_roughness(gray)
            
            return {
                "lbp_features": lbp_features,
                "texture_statistics": texture_stats,
                "detected_patterns": pattern_info,
                "surface_roughness": roughness,
                "texture_class": self._classify_texture(lbp_features, texture_stats)
            }
            
        except Exception as e:
            return {"error": f"Doku analizi hatası: {str(e)}"}
    
    def _calculate_lbp_features(self, gray_image: np.ndarray) -> Dict[str, Any]:
        """Local Binary Pattern özelliklerini hesaplar."""
        radius = self.config["lbp_radius"]
        n_points = self.config["lbp_n_points"]
        
        # LBP hesapla
        lbp = self._local_binary_pattern(gray_image, n_points, radius)
        
        # LBP histogramı
        hist, _ = np.histogram(lbp.ravel(), bins=n_points + 2, range=(0, n_points + 2))
        hist = hist.astype(float)
        hist /= (hist.sum() + 1e-7)  # Normalize
        
        # Uniform patterns sayısı
        uniform_patterns = self._count_uniform_patterns(lbp, n_points)
        
        return {
            "histogram": hist.tolist(),
            "uniform_patterns": uniform_patterns,
            "energy": float(np.sum(hist ** 2)),
            "entropy": float(-np.sum(hist * np.log2(hist + 1e-7))),
            "contrast": float(np.var(lbp))
        }
    
    def _local_binary_pattern(self, image: np.ndarray, n_points: int, radius: float) -> np.ndarray:
        """Local Binary Pattern hesaplar."""
        # Basit LBP implementasyonu
        rows, cols = image.shape
        lbp = np.zeros_like(image)
        
        # Çember üzerindeki noktalar
        angles = 2 * np.pi * np.arange(n_points) / n_points
        dx = radius * np.cos(angles)
        dy = -radius * np.sin(angles)
        
        for i in range(radius, rows - radius):
            for j in range(radius, cols - radius):
                center_value = image[i, j]
                pattern = 0
                
                for k in range(n_points):
                    # Komşu piksel koordinatları
                    x = i + int(round(dy[k]))
                    y = j + int(round(dx[k]))
                    
                    # Bilinear interpolation
                    if 0 <= x < rows and 0 <= y < cols:
                        neighbor_value = image[x, y]
                        if neighbor_value >= center_value:
                            pattern |= (1 << k)
                
                lbp[i, j] = pattern
        
        return lbp
    
    def _count_uniform_patterns(self, lbp: np.ndarray, n_points: int) -> int:
        """Uniform pattern sayısını hesaplar."""
        uniform_count = 0
        
        for value in np.unique(lbp):
            # Binary representation
            binary = format(int(value), f'0{n_points}b')
            
            # Transition sayısını hesapla
            transitions = 0
            for i in range(n_points):
                if binary[i] != binary[(i + 1) % n_points]:
                    transitions += 1
            
            # Uniform pattern: en fazla 2 transition
            if transitions <= 2:
                uniform_count += np.sum(lbp == value)
        
        return uniform_count
    
    def _calculate_texture_statistics(self, gray_image: np.ndarray) -> Dict[str, Any]:
        """Temel doku istatistiklerini hesaplar."""
        # Gradyan hesapla
        grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Laplacian (ikinci türev)
        laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
        
        return {
            "mean_intensity": float(np.mean(gray_image)),
            "std_intensity": float(np.std(gray_image)),
            "gradient_mean": float(np.mean(gradient_magnitude)),
            "gradient_std": float(np.std(gradient_magnitude)),
            "laplacian_variance": float(np.var(laplacian)),
            "homogeneity": float(1.0 / (1.0 + np.var(gray_image))),
            "contrast": float(np.std(gray_image) / (np.mean(gray_image) + 1e-7))
        }
    
    def _detect_patterns(self, gray_image: np.ndarray) -> Dict[str, Any]:
        """Yüzey desenlerini tespit eder."""
        if not self.config["pattern_detection"]:
            return {"detected": False}
        
        patterns = {}
        
        # Çizgili desen tespiti (Hough Lines)
        edges = cv2.Canny(gray_image, 50, 150)
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        
        if lines is not None:
            # Paralel çizgileri say
            angles = []
            for line in lines:
                rho, theta = line[0]
                angles.append(theta)
            
            # Benzer açılardaki çizgileri grupla
            angles = np.array(angles)
            angle_groups = []
            for angle in np.unique(np.round(angles, 1)):
                similar_angles = angles[np.abs(angles - angle) < 0.1]
                if len(similar_angles) >= 3:  # En az 3 paralel çizgi
                    angle_groups.append(len(similar_angles))
            
            if angle_groups:
                patterns["striped"] = {
                    "detected": True,
                    "line_groups": len(angle_groups),
                    "max_parallel_lines": max(angle_groups)
                }
        
        # Noktalı desen tespiti (Hough Circles)
        circles = cv2.HoughCircles(
            gray_image,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=20,
            param1=50,
            param2=30,
            minRadius=3,
            maxRadius=15
        )
        
        if circles is not None:
            patterns["dotted"] = {
                "detected": True,
                "dot_count": len(circles[0])
            }
        
        # Dama tahtası deseni
        checkerboard_score = self._detect_checkerboard(gray_image)
        if checkerboard_score > 0.3:
            patterns["checkerboard"] = {
                "detected": True,
                "score": checkerboard_score
            }
        
        return patterns if patterns else {"detected": False}
    
    def _detect_checkerboard(self, gray_image: np.ndarray) -> float:
        """Dama tahtası deseni tespit eder."""
        # Template matching ile dama tahtası tespiti
        h, w = gray_image.shape
        
        # Küçük dama tahtası template'i oluştur
        template_size = min(20, h//4, w//4)
        if template_size < 8:
            return 0.0
        
        template = np.zeros((template_size, template_size), dtype=np.uint8)
        half = template_size // 2
        template[:half, :half] = 255
        template[half:, half:] = 255
        
        # Template matching
        result = cv2.matchTemplate(gray_image, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        
        return float(max_val)
    
    def _analyze_surface_roughness(self, gray_image: np.ndarray) -> Dict[str, Any]:
        """Yüzey pürüzlülüğünü analiz eder."""
        # Farklı kernel boyutlarında Laplacian variance
        roughness_scales = {}
        
        for kernel_size in [3, 5, 7]:
            laplacian = cv2.Laplacian(gray_image, cv2.CV_64F, ksize=kernel_size)
            variance = np.var(laplacian)
            roughness_scales[f"scale_{kernel_size}"] = float(variance)
        
        # Genel pürüzlülük skoru
        avg_roughness = np.mean(list(roughness_scales.values()))
        
        # Pürüzlülük sınıfı
        if avg_roughness < 100:
            roughness_class = "smooth"
        elif avg_roughness < 500:
            roughness_class = "medium"
        else:
            roughness_class = "rough"
        
        return {
            "class": roughness_class,
            "score": float(avg_roughness),
            "scales": roughness_scales
        }
    
    def _classify_texture(self, lbp_features: Dict, texture_stats: Dict) -> str:
        """Doku tipini sınıflandırır."""
        # Basit kural tabanlı sınıflandırma
        
        # Uniform pattern oranı
        total_pixels = lbp_features.get("uniform_patterns", 0)
        energy = lbp_features.get("energy", 0)
        contrast = texture_stats.get("contrast", 0)
        
        if energy > 0.1 and contrast < 0.3:
            return "smooth"
        elif energy < 0.05 and contrast > 0.7:
            return "rough"
        elif 0.05 <= energy <= 0.1:
            return "medium"
        else:
            return "unknown"
