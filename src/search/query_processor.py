"""
QueryProcessor - Sorgu ön işleme modülü
Sorgu verilerini temizler, doğrular ve optimize eder.
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Optional


class QueryProcessor:
    """
    Sorgu verilerini ön işleyen sınıf.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        QueryProcessor constructor
        
        Args:
            config: Sorgu işleme konfigürasyonu
        """
        self.config = config or {
            'confidence_threshold': 0.3,
            'feature_validation': True,
            'preprocessing_enabled': True
        }
    
    def process_query(self, query_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sorgu verilerini işler ve optimize eder.
        
        Args:
            query_data: Ham sorgu verileri
            
        Returns:
            İşlenmiş sorgu verileri
        """
        try:
            processed_query = query_data.copy()
            
            if self.config.get('preprocessing_enabled', True):
                # Güven skoru kontrolü
                processed_query = self._filter_by_confidence(processed_query)
                
                # Özellik doğrulama
                if self.config.get('feature_validation', True):
                    processed_query = self._validate_features(processed_query)
                
                # Normalizasyon
                processed_query = self._normalize_features(processed_query)
                
                # Eksik özellikleri tamamla
                processed_query = self._fill_missing_features(processed_query)
            
            return processed_query
            
        except Exception as e:
            print(f"Sorgu işleme hatası: {str(e)}")
            return query_data
    
    def _filter_by_confidence(self, query_data: Dict[str, Any]) -> Dict[str, Any]:
        """Güven skoruna göre filtreler."""
        confidence_scores = query_data.get('confidence_scores', {})
        threshold = self.config['confidence_threshold']
        
        # Düşük güvenli bileşenleri işaretle
        filtered_data = query_data.copy()
        
        for component, score in confidence_scores.items():
            if score < threshold:
                # Düşük güvenli bileşeni işaretle
                if 'low_confidence_components' not in filtered_data:
                    filtered_data['low_confidence_components'] = []
                filtered_data['low_confidence_components'].append(component)
        
        return filtered_data
    
    def _validate_features(self, query_data: Dict[str, Any]) -> Dict[str, Any]:
        """Özellik verilerini doğrular."""
        validated_data = query_data.copy()
        
        # Renk özelliklerini doğrula
        if 'color_analysis' in validated_data:
            validated_data['color_analysis'] = self._validate_color_features(
                validated_data['color_analysis']
            )
        
        # Şekil özelliklerini doğrula
        if 'form_size' in validated_data:
            validated_data['form_size'] = self._validate_shape_features(
                validated_data['form_size']
            )
        
        # SIFT özelliklerini doğrula
        if 'visual_features' in validated_data:
            validated_data['visual_features'] = self._validate_visual_features(
                validated_data['visual_features']
            )
        
        return validated_data
    
    def _validate_color_features(self, color_data: Dict[str, Any]) -> Dict[str, Any]:
        """Renk özelliklerini doğrular."""
        validated = color_data.copy()
        
        # Dominant renkler kontrolü
        if 'dominant_colors' in validated:
            valid_colors = []
            for color in validated['dominant_colors']:
                if self._is_valid_color(color):
                    valid_colors.append(color)
            validated['dominant_colors'] = valid_colors
        
        return validated
    
    def _is_valid_color(self, color: Dict[str, Any]) -> bool:
        """Renk verisinin geçerliliğini kontrol eder."""
        # RGB değerleri kontrolü
        rgb = color.get('rgb', [])
        if len(rgb) != 3:
            return False
        
        for value in rgb:
            if not isinstance(value, (int, float)) or value < 0 or value > 255:
                return False
        
        # Yüzde kontrolü
        percentage = color.get('percentage', 0)
        if not isinstance(percentage, (int, float)) or percentage < 0 or percentage > 1:
            return False
        
        return True
    
    def _validate_shape_features(self, shape_data: Dict[str, Any]) -> Dict[str, Any]:
        """Şekil özelliklerini doğrular."""
        validated = shape_data.copy()
        
        # Bagaj tipi kontrolü
        valid_types = ['hardshell', 'softshell', 'duffel', 'backpack', 'unknown']
        if validated.get('suitcase_type') not in valid_types:
            validated['suitcase_type'] = 'unknown'
        
        # Boyut sınıfı kontrolü
        valid_sizes = ['small', 'medium', 'large', 'oversized', 'unknown']
        if validated.get('size_class') not in valid_sizes:
            validated['size_class'] = 'unknown'
        
        # Tekerlek sayısı kontrolü
        wheel_count = validated.get('wheel_count', 0)
        if not isinstance(wheel_count, int) or wheel_count < 0 or wheel_count > 8:
            validated['wheel_count'] = 0
        
        return validated
    
    def _validate_visual_features(self, visual_data: Dict[str, Any]) -> Dict[str, Any]:
        """Görsel özelliklerini doğrular."""
        validated = visual_data.copy()
        
        # SIFT keypoints kontrolü
        if 'sift_keypoints' in validated:
            valid_keypoints = []
            for kp in validated['sift_keypoints']:
                if self._is_valid_keypoint(kp):
                    valid_keypoints.append(kp)
            validated['sift_keypoints'] = valid_keypoints
        
        # Global embedding kontrolü
        if 'global_embedding' in validated:
            embedding = validated['global_embedding']
            if not isinstance(embedding, list) or len(embedding) == 0:
                validated['global_embedding'] = []
            else:
                # NaN değerleri temizle
                cleaned_embedding = []
                for val in embedding:
                    if isinstance(val, (int, float)) and not np.isnan(val):
                        cleaned_embedding.append(float(val))
                validated['global_embedding'] = cleaned_embedding
        
        return validated
    
    def _is_valid_keypoint(self, keypoint: Dict[str, Any]) -> bool:
        """Keypoint verisinin geçerliliğini kontrol eder."""
        required_fields = ['x', 'y', 'size', 'angle', 'response', 'descriptor']
        
        for field in required_fields:
            if field not in keypoint:
                return False
        
        # Koordinat kontrolü
        x, y = keypoint.get('x', 0), keypoint.get('y', 0)
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            return False
        
        # Descriptor kontrolü
        descriptor = keypoint.get('descriptor', [])
        if not isinstance(descriptor, list) or len(descriptor) != 128:  # SIFT descriptor size
            return False
        
        return True
    
    def _normalize_features(self, query_data: Dict[str, Any]) -> Dict[str, Any]:
        """Özellikleri normalize eder."""
        normalized = query_data.copy()
        
        # Renk histogramlarını normalize et
        if 'visual_features' in normalized and 'histograms' in normalized['visual_features']:
            histograms = normalized['visual_features']['histograms']
            for hist_name, hist_data in histograms.items():
                if isinstance(hist_data, list) and hist_data:
                    total = sum(hist_data)
                    if total > 0:
                        normalized_hist = [val / total for val in hist_data]
                        normalized['visual_features']['histograms'][hist_name] = normalized_hist
        
        # Global embedding'i normalize et
        if 'visual_features' in normalized and 'global_embedding' in normalized['visual_features']:
            embedding = normalized['visual_features']['global_embedding']
            if embedding:
                embedding_array = np.array(embedding)
                norm = np.linalg.norm(embedding_array)
                if norm > 0:
                    normalized_embedding = (embedding_array / norm).tolist()
                    normalized['visual_features']['global_embedding'] = normalized_embedding
        
        return normalized
    
    def _fill_missing_features(self, query_data: Dict[str, Any]) -> Dict[str, Any]:
        """Eksik özellikleri varsayılan değerlerle doldurur."""
        completed = query_data.copy()
        
        # Eksik renk bilgileri
        if 'color_analysis' not in completed:
            completed['color_analysis'] = {
                'dominant_colors': [],
                'finish': 'unknown'
            }
        
        # Eksik şekil bilgileri
        if 'form_size' not in completed:
            completed['form_size'] = {
                'suitcase_type': 'unknown',
                'size_class': 'unknown',
                'wheel_count': 0
            }
        
        # Eksik görsel özellikler
        if 'visual_features' not in completed:
            completed['visual_features'] = {
                'sift_keypoints': [],
                'global_embedding': []
            }
        
        return completed
    
    def validate_query_schema(self, query_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Sorgu verilerinin şema uyumluluğunu kontrol eder.
        
        Returns:
            (is_valid, error_messages)
        """
        errors = []
        
        # Temel alan kontrolü
        if 'metadata' not in query_data:
            errors.append("Metadata eksik")
        
        # En az bir analiz bileşeni olmalı
        required_components = ['color_analysis', 'form_size', 'visual_features', 'texture_patterns']
        if not any(comp in query_data for comp in required_components):
            errors.append("En az bir analiz bileşeni gerekli")
        
        # Metadata kontrolü
        if 'metadata' in query_data:
            metadata = query_data['metadata']
            if 'source_filename' not in metadata:
                errors.append("Kaynak dosya adı eksik")
        
        return len(errors) == 0, errors
