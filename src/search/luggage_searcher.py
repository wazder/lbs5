"""
LuggageSearcher - Ana arama motoru
Benzerlik hesaplama ve eşleştirme fonksiyonlarını içerir.
"""

import numpy as np
import json
import os
from typing import Dict, List, Any, Tuple, Optional
from .similarity_calculator import SimilarityCalculator
from .query_processor import QueryProcessor
from .group_manager import GroupManager


class LuggageSearcher:
    """
    Bagaj arama ve eşleştirme motoru.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        LuggageSearcher constructor
        
        Args:
            config: Arama konfigürasyonu
        """
        self.config = config or self._get_default_config()
        
        # Alt bileşenler
        self.similarity_calculator = SimilarityCalculator(self.config.get('similarity', {}))
        self.query_processor = QueryProcessor(self.config.get('query', {}))
        self.group_manager = GroupManager(self.config.get('grouping', {}))
        
        # Galeri verileri
        self.gallery_data = {}
        self.gallery_loaded = False
    
    def load_gallery(self, gallery_path: str) -> bool:
        """
        Galeri verilerini yükler.
        
        Args:
            gallery_path: Galeri JSON dosyalarının bulunduğu klasör
            
        Returns:
            Yükleme başarı durumu
        """
        try:
            self.gallery_data = {}
            
            # JSON dosyalarını bul ve yükle
            for filename in os.listdir(gallery_path):
                if filename.endswith('.json'):
                    file_path = os.path.join(gallery_path, filename)
                    
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        
                        # Dosya adını key olarak kullan
                        image_id = os.path.splitext(filename)[0]
                        self.gallery_data[image_id] = data
            
            # Grup bazlı arama için kümeleme yap
            if self.config.get('use_grouping', True):
                self.group_manager.create_groups(self.gallery_data)
            
            self.gallery_loaded = True
            print(f"Galeri yüklendi: {len(self.gallery_data)} bagaj")
            
            return True
            
        except Exception as e:
            print(f"Galeri yükleme hatası: {str(e)}")
            return False
    
    def search(self, query_data: Dict[str, Any], 
              search_mode: str = 'global',
              top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Bagaj arama yapar.
        
        Args:
            query_data: Sorgu bagajın analiz verileri
            search_mode: Arama modu ('global', 'group_based', 'multi_image')
            top_k: Döndürülecek en iyi sonuç sayısı
            
        Returns:
            Benzerlik skorlarına göre sıralanmış sonuçlar
        """
        if not self.gallery_loaded:
            raise RuntimeError("Galeri yüklenmedi. Önce load_gallery() çağırın.")
        
        try:
            # Sorguyu ön işle
            processed_query = self.query_processor.process_query(query_data)
            
            # Arama moduna göre farklı stratejiler
            if search_mode == 'global':
                results = self._global_search(processed_query, top_k)
            elif search_mode == 'group_based':
                results = self._group_based_search(processed_query, top_k)
            elif search_mode == 'multi_image':
                results = self._multi_image_search(processed_query, top_k)
            else:
                raise ValueError(f"Desteklenmeyen arama modu: {search_mode}")
            
            return results
            
        except Exception as e:
            raise RuntimeError(f"Arama hatası: {str(e)}")
    
    def _global_search(self, query_data: Dict[str, Any], top_k: int) -> List[Dict[str, Any]]:
        """Tüm galeri üzerinde doğrudan arama."""
        similarities = []
        
        for gallery_id, gallery_item in self.gallery_data.items():
            # Benzerlik hesapla
            similarity_score = self.similarity_calculator.calculate_similarity(
                query_data, gallery_item
            )
            
            similarities.append({
                'gallery_id': gallery_id,
                'similarity_score': similarity_score['overall_score'],
                'component_scores': similarity_score['component_scores'],
                'gallery_metadata': gallery_item.get('metadata', {}),
                'matching_details': similarity_score.get('details', {})
            })
        
        # Benzerlik skoruna göre sırala
        similarities.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        return similarities[:top_k]
    
    def _group_based_search(self, query_data: Dict[str, Any], top_k: int) -> List[Dict[str, Any]]:
        """Grup bazlı hızlandırılmış arama."""
        if not hasattr(self.group_manager, 'groups'):
            # Grup yoksa global arama yap
            return self._global_search(query_data, top_k)
        
        # En yakın grupları bul
        candidate_groups = self.group_manager.find_candidate_groups(query_data)
        
        similarities = []
        
        # Sadece aday gruplardaki öğeleri ara
        for group_id in candidate_groups:
            group_items = self.group_manager.get_group_items(group_id)
            
            for gallery_id in group_items:
                if gallery_id in self.gallery_data:
                    gallery_item = self.gallery_data[gallery_id]
                    
                    similarity_score = self.similarity_calculator.calculate_similarity(
                        query_data, gallery_item
                    )
                    
                    similarities.append({
                        'gallery_id': gallery_id,
                        'similarity_score': similarity_score['overall_score'],
                        'component_scores': similarity_score['component_scores'],
                        'gallery_metadata': gallery_item.get('metadata', {}),
                        'group_id': group_id,
                        'matching_details': similarity_score.get('details', {})
                    })
        
        # Benzerlik skoruna göre sırala
        similarities.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        return similarities[:top_k]
    
    def _multi_image_search(self, query_data_list: List[Dict[str, Any]], 
                           top_k: int) -> List[Dict[str, Any]]:
        """Çoklu görüntü sorgusu."""
        if not isinstance(query_data_list, list):
            query_data_list = [query_data_list]
        
        # Her sorgu için sonuçları topla
        all_similarities = {}
        
        for query_data in query_data_list:
            results = self._global_search(query_data, len(self.gallery_data))
            
            for result in results:
                gallery_id = result['gallery_id']
                
                if gallery_id not in all_similarities:
                    all_similarities[gallery_id] = {
                        'gallery_id': gallery_id,
                        'scores': [],
                        'gallery_metadata': result['gallery_metadata']
                    }
                
                all_similarities[gallery_id]['scores'].append(result['similarity_score'])
        
        # Skorları birleştir (ortalama)
        final_results = []
        for gallery_id, data in all_similarities.items():
            avg_score = np.mean(data['scores'])
            max_score = np.max(data['scores'])
            
            final_results.append({
                'gallery_id': gallery_id,
                'similarity_score': avg_score,
                'max_score': max_score,
                'num_matches': len(data['scores']),
                'gallery_metadata': data['gallery_metadata']
            })
        
        # Ortalama skora göre sırala
        final_results.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        return final_results[:top_k]
    
    def batch_search(self, query_folder: str, output_folder: str,
                    search_mode: str = 'global', top_k: int = 10) -> Dict[str, Any]:
        """
        Toplu arama işlemi.
        
        Args:
            query_folder: Sorgu JSON dosyalarının klasörü
            output_folder: Sonuçların kaydedileceği klasör
            search_mode: Arama modu
            top_k: Her sorgu için döndürülecek sonuç sayısı
            
        Returns:
            Toplu işlem istatistikleri
        """
        if not self.gallery_loaded:
            raise RuntimeError("Galeri yüklenmedi.")
        
        # Output klasörünü oluştur
        os.makedirs(output_folder, exist_ok=True)
        
        results_summary = {
            'total_queries': 0,
            'successful_searches': 0,
            'failed_searches': 0,
            'average_processing_time': 0.0
        }
        
        import time
        total_time = 0.0
        
        # Query dosyalarını işle
        for filename in os.listdir(query_folder):
            if filename.endswith('.json'):
                query_file = os.path.join(query_folder, filename)
                
                try:
                    start_time = time.time()
                    
                    # Query verilerini yükle
                    with open(query_file, 'r', encoding='utf-8') as f:
                        query_data = json.load(f)
                    
                    # Arama yap
                    search_results = self.search(query_data, search_mode, top_k)
                    
                    # Sonucu kaydet
                    output_file = os.path.join(output_folder, f"result_{filename}")
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump({
                            'query_file': filename,
                            'search_mode': search_mode,
                            'top_k': top_k,
                            'results': search_results,
                            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                        }, f, indent=2, ensure_ascii=False)
                    
                    end_time = time.time()
                    total_time += (end_time - start_time)
                    
                    results_summary['successful_searches'] += 1
                    
                except Exception as e:
                    print(f"Arama hatası {filename}: {str(e)}")
                    results_summary['failed_searches'] += 1
                
                results_summary['total_queries'] += 1
        
        if results_summary['total_queries'] > 0:
            results_summary['average_processing_time'] = total_time / results_summary['total_queries']
        
        # Özet raporu kaydet
        summary_file = os.path.join(output_folder, 'batch_summary.json')
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(results_summary, f, indent=2)
        
        return results_summary
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Varsayılan arama konfigürasyonu."""
        return {
            'similarity': {
                'weights': {
                    'color': 0.4,
                    'shape': 0.3,
                    'texture': 0.2,
                    'sift': 0.1
                }
            },
            'query': {
                'confidence_threshold': 0.3,
                'feature_validation': True
            },
            'grouping': {
                'enabled': True,
                'num_clusters': 10,
                'features_for_clustering': ['color', 'shape']
            },
            'use_grouping': True
        }
