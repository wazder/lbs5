"""
FeatureExtractor - Görsel özellik çıkarımı
SIFT keypoints, global embedding ve diğer görsel özellikleri çıkarır.
"""

import cv2
import numpy as np
from typing import Dict, List, Any, Optional


class FeatureExtractor:
    """
    Görsel özellik çıkarımı için sınıf.
    """
    
    def __init__(self):
        """FeatureExtractor constructor"""
        # SIFT detector'ı başlat
        self.sift = cv2.SIFT_create()
        
        # ORB detector (alternatif)
        self.orb = cv2.ORB_create()
    
    def extract_features(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Görsel özellikleri çıkarır.
        Sistem artık sadece en detaylı modu kullanır.
        
        Args:
            image: Analiz edilecek resim
            
        Returns:
            Görsel özellikleri içeren sözlük
        """
        features = {}
        
        max_sift_features = 50
        embedding_enabled = True
        
        # SIFT keypoints
        features["sift_keypoints"] = self._extract_sift_features(
            image, max_features=max_sift_features
        )
        
        # Global embedding
        if embedding_enabled:
            features["global_embedding"] = self._extract_global_embedding(image)
        
        # Color moments
        features["color_moments"] = self._calculate_color_moments(image)
        
        # Histogram features
        features["histograms"] = self._extract_histograms(image)
        
        return features
    
    def _extract_sift_features(self, image: np.ndarray, 
                              max_features: int = 20) -> List[Dict[str, Any]]:
        """SIFT keypoint'lerini çıkarır."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # SIFT detector ile keypoint ve descriptor bul
        keypoints, descriptors = self.sift.detectAndCompute(gray, None)
        
        if descriptors is None:
            return []
        
        # En güçlü max_features kadar keypoint seç
        if len(keypoints) > max_features:
            # Response değerine göre sırala
            keypoints_with_response = [(kp, desc) for kp, desc in zip(keypoints, descriptors)]
            keypoints_with_response.sort(key=lambda x: x[0].response, reverse=True)
            keypoints = [kp for kp, _ in keypoints_with_response[:max_features]]
            descriptors = np.array([desc for _, desc in keypoints_with_response[:max_features]])
        
        # Keypoint bilgilerini serialize et
        sift_features = []
        for i, kp in enumerate(keypoints):
            feature = {
                "x": float(kp.pt[0]),
                "y": float(kp.pt[1]),
                "size": float(kp.size),
                "angle": float(kp.angle),
                "response": float(kp.response),
                "octave": int(kp.octave),
                "descriptor": descriptors[i].tolist()
            }
            sift_features.append(feature)
        
        return sift_features
    
    def _extract_global_embedding(self, image: np.ndarray) -> List[float]:
        """Global görsel embedding çıkarır."""
        # Basit global feature: resized image'ın düzleştirilmiş hali
        # Gerçek uygulamada CNN features kullanılabilir
        
        # Küçük boyuta getir
        resized = cv2.resize(image, (64, 64))
        
        # Normalize et
        normalized = resized.astype(np.float32) / 255.0
        
        # Düzleştir
        flattened = normalized.flatten()
        
        # PCA benzeri boyut azaltma (basit)
        # İlk 512 component al
        if len(flattened) > 512:
            step = len(flattened) // 512
            embedding = flattened[::step][:512]
        else:
            embedding = flattened
        
        return embedding.tolist()
    
    def _calculate_color_moments(self, image: np.ndarray) -> Dict[str, List[float]]:
        """Renk momentlerini hesaplar."""
        # BGR kanallarını ayır
        b, g, r = cv2.split(image)
        
        moments = {}
        
        for channel_name, channel in [('blue', b), ('green', g), ('red', r)]:
            # Her kanal için momentler
            mean = float(np.mean(channel))
            std = float(np.std(channel))
            skewness = float(self._calculate_skewness(channel))
            
            moments[channel_name] = {
                "mean": mean,
                "std": std,
                "skewness": skewness
            }
        
        return moments
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Skewness (çarpıklık) hesaplar."""
        mean = np.mean(data)
        std = np.std(data)
        
        if std == 0:
            return 0.0
        
        normalized = (data - mean) / std
        skewness = np.mean(normalized ** 3)
        
        return skewness
    
    def _extract_histograms(self, image: np.ndarray) -> Dict[str, List[int]]:
        """Renk histogramlarını çıkarır."""
        histograms = {}
        
        # BGR histogramları
        for i, color in enumerate(['blue', 'green', 'red']):
            hist = cv2.calcHist([image], [i], None, [256], [0, 256])
            histograms[f'{color}_histogram'] = hist.flatten().astype(int).tolist()
        
        # HSV histogramları
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        for i, component in enumerate(['hue', 'saturation', 'value']):
            hist = cv2.calcHist([hsv], [i], None, [256], [0, 256])
            histograms[f'{component}_histogram'] = hist.flatten().astype(int).tolist()
        
        return histograms
    
    def extract_orb_features(self, image: np.ndarray, 
                           max_features: int = 50) -> List[Dict[str, Any]]:
        """ORB keypoint'lerini çıkarır (SIFT alternatifi)."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # ORB ile keypoint ve descriptor bul
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        
        if descriptors is None:
            return []
        
        # En güçlü max_features kadar keypoint seç
        if len(keypoints) > max_features:
            keypoints_with_response = [(kp, desc) for kp, desc in zip(keypoints, descriptors)]
            keypoints_with_response.sort(key=lambda x: x[0].response, reverse=True)
            keypoints = [kp for kp, _ in keypoints_with_response[:max_features]]
            descriptors = np.array([desc for _, desc in keypoints_with_response[:max_features]])
        
        # Keypoint bilgilerini serialize et
        orb_features = []
        for i, kp in enumerate(keypoints):
            feature = {
                "x": float(kp.pt[0]),
                "y": float(kp.pt[1]),
                "size": float(kp.size),
                "angle": float(kp.angle),
                "response": float(kp.response),
                "descriptor": descriptors[i].tolist()
            }
            orb_features.append(feature)
        
        return orb_features
