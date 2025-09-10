"""
GroupManager - Grup bazlı arama optimizasyonu
Bagajları kümelere ayırarak arama performansını artırır.
"""

import numpy as np
import json
from typing import Dict, List, Any, Optional
from sklearn.cluster import KMeans


class GroupManager:
    """
    Bagajları gruplara ayıran ve grup bazlı arama sağlayan sınıf.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        GroupManager constructor
        
        Args:
            config: Gruplama konfigürasyonu
        """
        self.config = config or {
            'enabled': True,
            'num_clusters': 10,
            'features_for_clustering': ['color', 'shape'],
            'cluster_algorithm': 'kmeans'
        }
        
        self.groups = {}
        self.group_centers = {}
        self.feature_vectors = {}
        self.item_to_group = {}
    
    def create_groups(self, gallery_data: Dict[str, Any]) -> bool:
        """
        Galeri verilerinden gruplar oluşturur.
        
        Args:
            gallery_data: Galeri verileri
            
        Returns:
            Gruplama başarı durumu
        """
        try:
            if not self.config.get('enabled', True):
                return False
            
            # Feature vektörleri çıkar
            feature_vectors = self._extract_feature_vectors(gallery_data)
            
            if len(feature_vectors) < self.config['num_clusters']:
                print(f"Yeterli veri yok: {len(feature_vectors)} < {self.config['num_clusters']}")
                return False
            
            # Kümeleme yap
            cluster_labels = self._perform_clustering(feature_vectors)
            
            # Grupları oluştur
            self._create_group_structure(feature_vectors, cluster_labels)
            
            print(f"Gruplama tamamlandı: {len(self.groups)} grup oluşturuldu")
            return True
            
        except Exception as e:
            print(f"Gruplama hatası: {str(e)}")
            return False
    
    def _extract_feature_vectors(self, gallery_data: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Her bagaj için özellik vektörü çıkarır."""
        feature_vectors = {}
        
        for item_id, item_data in gallery_data.items():
            try:
                vector = self._create_feature_vector(item_data)
                if vector is not None:
                    feature_vectors[item_id] = vector
                    self.feature_vectors[item_id] = vector
            except Exception as e:
                print(f"Özellik çıkarma hatası {item_id}: {str(e)}")
        
        return feature_vectors
    
    def _create_feature_vector(self, item_data: Dict[str, Any]) -> Optional[np.ndarray]:
        """Tek bir bagaj için özellik vektörü oluşturur."""
        features = []
        
        # Renk özellikleri
        if 'color' in self.config['features_for_clustering']:
            color_features = self._extract_color_features(item_data.get('color_analysis', {}))
            features.extend(color_features)
        
        # Şekil özellikleri
        if 'shape' in self.config['features_for_clustering']:
            shape_features = self._extract_shape_features(item_data.get('form_size', {}))
            features.extend(shape_features)
        
        # Doku özellikleri
        if 'texture' in self.config['features_for_clustering']:
            texture_features = self._extract_texture_features(item_data.get('texture_patterns', {}))
            features.extend(texture_features)
        
        if not features:
            return None
        
        return np.array(features)
    
    def _extract_color_features(self, color_data: Dict[str, Any]) -> List[float]:
        """Renk özelliklerini çıkarır."""
        features = []
        
        # Dominant renkler (RGB ortalama)
        dominant_colors = color_data.get('dominant_colors', [])
        if dominant_colors:
            # İlk 3 dominant rengin RGB değerleri
            for i in range(3):
                if i < len(dominant_colors):
                    rgb = dominant_colors[i].get('rgb', [0, 0, 0])
                    features.extend([val / 255.0 for val in rgb])  # Normalize
                else:
                    features.extend([0.0, 0.0, 0.0])  # Padding
        else:
            features.extend([0.0] * 9)  # 3 renk x 3 kanal
        
        # Finish tipi (surface properties)
        finish_types = ['matte', 'glossy', 'textured', 'unknown']
        finish = color_data.get('finish', 'unknown')
        finish_vector = [1.0 if fin == finish else 0.0 for fin in finish_types]
        features.extend(finish_vector)
        
        return features
    
    def _extract_shape_features(self, shape_data: Dict[str, Any]) -> List[float]:
        """Şekil özelliklerini çıkarır."""
        features = []
        
        # Bagaj tipi (one-hot encoding)
        suitcase_types = ['hardshell', 'softshell', 'duffel', 'backpack', 'unknown']
        suitcase_type = shape_data.get('suitcase_type', 'unknown')
        type_vector = [1.0 if typ == suitcase_type else 0.0 for typ in suitcase_types]
        features.extend(type_vector)
        
        # Boyut sınıfı (one-hot encoding)
        size_classes = ['small', 'medium', 'large', 'oversized']
        size_class = shape_data.get('size_class', 'medium')
        if size_class not in size_classes:
            size_class = 'medium'
        size_vector = [1.0 if size == size_class else 0.0 for size in size_classes]
        features.extend(size_vector)
        
        # Tekerlek sayısı (normalize)
        wheel_count = shape_data.get('wheel_count', 0)
        features.append(wheel_count / 8.0)  # Max 8 tekerlek varsayımı
        
        # Aspect ratio
        aspect_ratio = shape_data.get('aspect_ratio', 1.0)
        features.append(min(aspect_ratio / 3.0, 1.0))  # Normalize to [0,1]
        
        return features
    
    def _extract_texture_features(self, texture_data: Dict[str, Any]) -> List[float]:
        """Doku özelliklerini çıkarır."""
        features = []
        
        # LBP histogram özeti (ilk 10 bin)
        lbp_features = texture_data.get('lbp_features', {})
        histogram = lbp_features.get('histogram', [])
        
        if histogram:
            # İlk 10 bin'i al ve normalize et
            hist_summary = histogram[:10] if len(histogram) >= 10 else histogram + [0.0] * (10 - len(histogram))
            features.extend(hist_summary)
        else:
            features.extend([0.0] * 10)
        
        # Doku sınıfı (one-hot)
        texture_classes = ['smooth', 'medium', 'rough', 'unknown']
        texture_class = texture_data.get('texture_class', 'unknown')
        class_vector = [1.0 if cls == texture_class else 0.0 for cls in texture_classes]
        features.extend(class_vector)
        
        return features
    
    def _perform_clustering(self, feature_vectors: Dict[str, np.ndarray]) -> Dict[str, int]:
        """K-means kümeleme yapar."""
        # Vektörleri matrise dönüştür
        item_ids = list(feature_vectors.keys())
        vectors = np.array([feature_vectors[item_id] for item_id in item_ids])
        
        # K-means
        n_clusters = min(self.config['num_clusters'], len(item_ids))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(vectors)
        
        # Küme merkezlerini kaydet
        self.group_centers = {i: center for i, center in enumerate(kmeans.cluster_centers_)}
        
        # Item-label mapping
        item_labels = {}
        for item_id, label in zip(item_ids, cluster_labels):
            item_labels[item_id] = int(label)
        
        return item_labels
    
    def _create_group_structure(self, feature_vectors: Dict[str, np.ndarray], 
                               cluster_labels: Dict[str, int]):
        """Grup yapısını oluşturur."""
        self.groups = {}
        self.item_to_group = cluster_labels.copy()
        
        # Her küme için grup oluştur
        for item_id, group_id in cluster_labels.items():
            if group_id not in self.groups:
                self.groups[group_id] = []
            
            self.groups[group_id].append(item_id)
        
        # Grup istatistikleri
        for group_id, items in self.groups.items():
            print(f"Grup {group_id}: {len(items)} öğe")
    
    def find_candidate_groups(self, query_data: Dict[str, Any], 
                            max_groups: int = 3) -> List[int]:
        """Sorgu için en yakın grupları bulur."""
        if not self.groups or not self.group_centers:
            return list(self.groups.keys()) if self.groups else []
        
        try:
            # Query için özellik vektörü oluştur
            query_vector = self._create_feature_vector(query_data)
            
            if query_vector is None:
                return list(self.groups.keys())
            
            # Her grup merkezine uzaklık hesapla
            group_distances = {}
            
            for group_id, center in self.group_centers.items():
                distance = np.linalg.norm(query_vector - center)
                group_distances[group_id] = distance
            
            # En yakın grupları seç
            sorted_groups = sorted(group_distances.items(), key=lambda x: x[1])
            candidate_groups = [group_id for group_id, _ in sorted_groups[:max_groups]]
            
            return candidate_groups
            
        except Exception as e:
            print(f"Aday grup bulma hatası: {str(e)}")
            return list(self.groups.keys())
    
    def get_group_items(self, group_id: int) -> List[str]:
        """Belirtilen grubun öğelerini döndürür."""
        return self.groups.get(group_id, [])
    
    def get_item_group(self, item_id: str) -> Optional[int]:
        """Öğenin hangi grupta olduğunu döndürür."""
        return self.item_to_group.get(item_id)
    
    def save_groups(self, output_path: str) -> bool:
        """Grup yapısını dosyaya kaydeder."""
        try:
            group_data = {
                'groups': self.groups,
                'group_centers': {str(k): v.tolist() for k, v in self.group_centers.items()},
                'item_to_group': self.item_to_group,
                'config': self.config
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(group_data, f, indent=2)
            
            return True
            
        except Exception as e:
            print(f"Grup kaydetme hatası: {str(e)}")
            return False
    
    def load_groups(self, input_path: str) -> bool:
        """Grup yapısını dosyadan yükler."""
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                group_data = json.load(f)
            
            self.groups = group_data['groups']
            self.group_centers = {int(k): np.array(v) for k, v in group_data['group_centers'].items()}
            self.item_to_group = group_data['item_to_group']
            self.config.update(group_data.get('config', {}))
            
            return True
            
        except Exception as e:
            print(f"Grup yükleme hatası: {str(e)}")
            return False
