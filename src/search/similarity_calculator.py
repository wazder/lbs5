"""
SimilarityCalculator - Benzerlik hesaplama modülü
Ağırlıklı benzerlik skorları ve mesafe metrikleri hesaplar.
"""

import numpy as np
from typing import Dict, List, Any, Tuple
import math


class SimilarityCalculator:
    """
    Bagaj özellikleri arasında benzerlik hesaplayan sınıf.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        SimilarityCalculator constructor
        
        Args:
            config: Benzerlik hesaplama konfigürasyonu
        """
        self.config = config or self._get_default_config()
        self.weights = self.config.get('weights', {})
        self.distance_metric = self.config.get('distance_metric', 'cosine')
    
    def calculate_similarity(self, query_data: Dict[str, Any], 
                           gallery_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        İki bagaj arasında benzerlik hesaplar.
        
        Args:
            query_data: Sorgu bagajın özellikleri
            gallery_data: Galeri bagajın özellikleri
            
        Returns:
            Benzerlik skorları ve detaylar
        """
        try:
            component_scores = {}
            
            # Renk benzerliği
            if 'color_analysis' in query_data and 'color_analysis' in gallery_data:
                component_scores['color'] = self._calculate_color_similarity(
                    query_data['color_analysis'], 
                    gallery_data['color_analysis']
                )
            
            # Şekil benzerliği
            if 'form_size' in query_data and 'form_size' in gallery_data:
                component_scores['shape'] = self._calculate_shape_similarity(
                    query_data['form_size'], 
                    gallery_data['form_size']
                )
            
            # Doku benzerliği
            if 'texture_patterns' in query_data and 'texture_patterns' in gallery_data:
                component_scores['texture'] = self._calculate_texture_similarity(
                    query_data['texture_patterns'], 
                    gallery_data['texture_patterns']
                )
            
            # SIFT özellik benzerliği
            if 'visual_features' in query_data and 'visual_features' in gallery_data:
                component_scores['sift'] = self._calculate_sift_similarity(
                    query_data['visual_features'], 
                    gallery_data['visual_features']
                )
            
            # Genel skor hesapla
            overall_score = self._calculate_weighted_score(component_scores)
            
            return {
                'overall_score': overall_score,
                'component_scores': component_scores,
                'details': {
                    'weights_used': self.weights,
                    'distance_metric': self.distance_metric
                }
            }
            
        except Exception as e:
            return {
                'overall_score': 0.0,
                'component_scores': {},
                'error': f"Benzerlik hesaplama hatası: {str(e)}"
            }
    
    def _calculate_color_similarity(self, query_color: Dict, gallery_color: Dict) -> float:
        """Renk benzerliği hesaplar."""
        try:
            score = 0.0
            
            # Dominant renk benzerliği
            query_colors = query_color.get('dominant_colors', [])
            gallery_colors = gallery_color.get('dominant_colors', [])
            
            if query_colors and gallery_colors:
                color_score = self._compare_dominant_colors(query_colors, gallery_colors)
                score += color_score * 0.8
            
            # Yüzey özelliği benzerliği (finish)
            query_finish = query_color.get('finish', 'unknown')
            gallery_finish = gallery_color.get('finish', 'unknown')
            
            if query_finish != 'unknown' and gallery_finish != 'unknown':
                finish_score = 1.0 if query_finish == gallery_finish else 0.3
                score += finish_score * 0.2
            
            return min(1.0, score)
            
        except Exception:
            return 0.0
    
    def _compare_dominant_colors(self, colors1: List[Dict], colors2: List[Dict]) -> float:
        """Dominant renkleri karşılaştırır."""
        if not colors1 or not colors2:
            return 0.0
        
        # Her renk için en yakın eşleşmeyi bul
        total_similarity = 0.0
        
        for color1 in colors1[:3]:  # İlk 3 dominant renk
            best_match = 0.0
            rgb1 = np.array(color1.get('rgb', [0, 0, 0]))
            
            for color2 in colors2[:3]:
                rgb2 = np.array(color2.get('rgb', [0, 0, 0]))
                
                # Euclidean distance normaliz et
                distance = np.linalg.norm(rgb1 - rgb2)
                similarity = 1.0 - (distance / (255 * math.sqrt(3)))  # Max distance = 255*sqrt(3)
                
                best_match = max(best_match, similarity)
            
            # Yüzde ağırlığı ile çarp
            weight = color1.get('percentage', 0.0)
            total_similarity += best_match * weight
        
        return total_similarity
    
    def _calculate_shape_similarity(self, query_shape: Dict, gallery_shape: Dict) -> float:
        """Şekil benzerliği hesaplar."""
        try:
            score = 0.0
            
            # Bagaj tipi benzerliği
            query_type = query_shape.get('suitcase_type', 'unknown')
            gallery_type = gallery_shape.get('suitcase_type', 'unknown')
            
            if query_type != 'unknown' and gallery_type != 'unknown':
                type_score = 1.0 if query_type == gallery_type else 0.3  # Kısmi skor
                score += type_score * 0.4
            
            # Boyut sınıfı benzerliği
            query_size = query_shape.get('size_class', 'unknown')
            gallery_size = gallery_shape.get('size_class', 'unknown')
            
            if query_size != 'unknown' and gallery_size != 'unknown':
                size_score = self._compare_size_classes(query_size, gallery_size)
                score += size_score * 0.3
            
            # Tekerlek sayısı benzerliği
            query_wheels = query_shape.get('wheel_count', 0)
            gallery_wheels = gallery_shape.get('wheel_count', 0)
            
            if query_wheels > 0 and gallery_wheels > 0:
                wheel_score = 1.0 if query_wheels == gallery_wheels else 0.5
                score += wheel_score * 0.3
            
            return min(1.0, score)
            
        except Exception:
            return 0.0
    
    def _compare_size_classes(self, size1: str, size2: str) -> float:
        """Boyut sınıfları arası benzerlik."""
        size_order = ['small', 'medium', 'large', 'oversized']
        
        try:
            idx1 = size_order.index(size1)
            idx2 = size_order.index(size2)
            
            # Komşu boyutlara kısmi skor ver
            distance = abs(idx1 - idx2)
            if distance == 0:
                return 1.0
            elif distance == 1:
                return 0.7
            elif distance == 2:
                return 0.3
            else:
                return 0.0
                
        except ValueError:
            return 0.0
    
    def _calculate_texture_similarity(self, query_texture: Dict, gallery_texture: Dict) -> float:
        """Doku benzerliği hesaplar."""
        try:
            score = 0.0
            
            # LBP histogram benzerliği
            query_lbp = query_texture.get('lbp_features', {})
            gallery_lbp = gallery_texture.get('lbp_features', {})
            
            if query_lbp and gallery_lbp:
                lbp_score = self._compare_histograms(
                    query_lbp.get('histogram', []),
                    gallery_lbp.get('histogram', [])
                )
                score += lbp_score * 0.6
            
            # Doku sınıfı benzerliği
            query_class = query_texture.get('texture_class', 'unknown')
            gallery_class = gallery_texture.get('texture_class', 'unknown')
            
            if query_class != 'unknown' and gallery_class != 'unknown':
                class_score = 1.0 if query_class == gallery_class else 0.0
                score += class_score * 0.4
            
            return min(1.0, score)
            
        except Exception:
            return 0.0
    
    def _calculate_sift_similarity(self, query_visual: Dict, gallery_visual: Dict) -> float:
        """SIFT özellik benzerliği hesaplar."""
        try:
            score = 0.0
            
            # SIFT keypoint matching
            query_sift = query_visual.get('sift_keypoints', [])
            gallery_sift = gallery_visual.get('sift_keypoints', [])
            
            if query_sift and gallery_sift:
                sift_score = self._match_sift_keypoints(query_sift, gallery_sift)
                score += sift_score * 0.7
            
            # Global embedding benzerliği
            query_embedding = query_visual.get('global_embedding', [])
            gallery_embedding = gallery_visual.get('global_embedding', [])
            
            if query_embedding and gallery_embedding:
                embedding_score = self._compare_embeddings(query_embedding, gallery_embedding)
                score += embedding_score * 0.3
            
            return min(1.0, score)
            
        except Exception:
            return 0.0
    
    def _match_sift_keypoints(self, keypoints1: List[Dict], keypoints2: List[Dict]) -> float:
        """SIFT keypoint eşleştirmesi yapar."""
        if not keypoints1 or not keypoints2:
            return 0.0
        
        try:
            # Descriptor'ları çıkar
            desc1 = np.array([kp['descriptor'] for kp in keypoints1])
            desc2 = np.array([kp['descriptor'] for kp in keypoints2])
            
            # Brute force matching
            matches = []
            
            for i, d1 in enumerate(desc1):
                best_distance = float('inf')
                second_best = float('inf')
                
                for j, d2 in enumerate(desc2):
                    distance = np.linalg.norm(d1 - d2)
                    
                    if distance < best_distance:
                        second_best = best_distance
                        best_distance = distance
                    elif distance < second_best:
                        second_best = distance
                
                # Lowe's ratio test
                if best_distance < 0.7 * second_best:
                    matches.append(best_distance)
            
            # Match quality skoru
            if matches:
                avg_distance = np.mean(matches)
                match_ratio = len(matches) / min(len(keypoints1), len(keypoints2))
                
                # Normalize distance (0-1 arası)
                distance_score = max(0, 1.0 - (avg_distance / 256.0))  # SIFT descriptor max distance
                
                return (distance_score * 0.7) + (match_ratio * 0.3)
            else:
                return 0.0
                
        except Exception:
            return 0.0
    
    def _compare_embeddings(self, emb1: List[float], emb2: List[float]) -> float:
        """Global embedding'leri karşılaştırır."""
        try:
            if len(emb1) != len(emb2):
                # Farklı boyutları hizala
                min_len = min(len(emb1), len(emb2))
                emb1 = emb1[:min_len]
                emb2 = emb2[:min_len]
            
            vec1 = np.array(emb1)
            vec2 = np.array(emb2)
            
            if self.distance_metric == 'cosine':
                # Cosine similarity
                dot_product = np.dot(vec1, vec2)
                norm1 = np.linalg.norm(vec1)
                norm2 = np.linalg.norm(vec2)
                
                if norm1 == 0 or norm2 == 0:
                    return 0.0
                
                cosine_sim = dot_product / (norm1 * norm2)
                return (cosine_sim + 1) / 2  # [-1,1] -> [0,1]
            
            elif self.distance_metric == 'euclidean':
                # Euclidean distance
                distance = np.linalg.norm(vec1 - vec2)
                max_distance = np.sqrt(len(emb1))  # Maksimum olası mesafe
                return 1.0 - (distance / max_distance)
            
            else:
                return 0.0
                
        except Exception:
            return 0.0
    
    def _compare_histograms(self, hist1: List[float], hist2: List[float]) -> float:
        """Histogram benzerliği hesaplar."""
        try:
            if not hist1 or not hist2 or len(hist1) != len(hist2):
                return 0.0
            
            h1 = np.array(hist1)
            h2 = np.array(hist2)
            
            # Chi-square distance
            chi_square = np.sum((h1 - h2) ** 2 / (h1 + h2 + 1e-10))
            
            # Normalize to [0,1]
            similarity = np.exp(-chi_square / 2)
            
            return similarity
            
        except Exception:
            return 0.0
    
    def _calculate_weighted_score(self, component_scores: Dict[str, float]) -> float:
        """Ağırlıklı genel skor hesaplar."""
        total_weight = 0.0
        weighted_sum = 0.0
        
        for component, score in component_scores.items():
            weight = self.weights.get(component, 0.0)
            weighted_sum += score * weight
            total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        return weighted_sum / total_weight
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Varsayılan konfigürasyon."""
        return {
            'weights': {
                'color': 0.4,
                'shape': 0.3,
                'texture': 0.2,
                'sift': 0.1
            },
            'distance_metric': 'cosine',
            'normalization': 'min_max'
        }
