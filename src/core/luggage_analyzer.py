"""
LuggageAnalyzer - Ana analiz motoru
Bagaj fotoğraflarından çok boyutlu özellik çıkarımı yapar.
"""

import cv2
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
import json

from ..analyzers.color_analyzer import ColorAnalyzer
from ..analyzers.shape_analyzer import ShapeAnalyzer
from ..analyzers.texture_analyzer import TextureAnalyzer
from ..utils.image_processor import ImageProcessor
from ..utils.feature_extractor import FeatureExtractor
from ..mask.mask_processor import MaskProcessor


class LuggageAnalyzer:
    """
    Ana bagaj analiz sınıfı.
    Renk, şekil, doku ve görsel özellik analizlerini koordine eder.
    """
    
    def __init__(self, config: Optional[Dict] = None, enable_masking: bool = True):
        """
        LuggageAnalyzer constructor
        
        Args:
            config: Konfigürasyon parametreleri
            enable_masking: Maskeleme sistemini etkinleştir
        """
        self.config = config or self._get_default_config()
        self.version = "1.0.0"
        self.enable_masking = enable_masking
        
        # Alt analizörleri başlat
        self.color_analyzer = ColorAnalyzer(self.config.get('color', {}))
        self.shape_analyzer = ShapeAnalyzer(self.config.get('shape', {}))
        self.texture_analyzer = TextureAnalyzer(self.config.get('texture', {}))
        
        # Yardımcı sınıflar
        self.image_processor = ImageProcessor()
        self.feature_extractor = FeatureExtractor()
        
        # Maskeleme sistemi
        if self.enable_masking:
            try:
                from ..mask.mask_processor import MaskProcessor
                masking_config = self.config.get('masking', {})
                if masking_config.get('enabled', True):
                    self.mask_processor = MaskProcessor(
                        sam_config=masking_config.get('sam_config', {})
                    )
                else:
                    self.mask_processor = None
            except ImportError:
                self.mask_processor = None
        else:
            self.mask_processor = None
    
    def analyze_image(self, image_path: str, use_masking: bool = None) -> Dict[str, Any]:
        """
        Tek bir bagaj resmini analiz eder.
        
        Args:
            image_path: Analiz edilecek resmin yolu
            use_masking: Maskeleme kullan (None=otomatik, True=zorla, False=kullanma)
            
        Returns:
            Analiz sonuçlarını içeren sözlük
        """
        try:
            # Maskeleme karar verme
            should_mask = use_masking
            if should_mask is None:
                should_mask = self.mask_processor is not None
                
            # Maskeleme işlemi
            if should_mask and self.mask_processor:
                # Maskelenmiş resmi al veya oluştur
                processed_image_path = self._get_or_create_masked_image(image_path)
                if processed_image_path:
                    image_path = processed_image_path
            
            # Resmi yükle ve ön işle
            image = self.image_processor.load_and_preprocess(image_path)
            
            if image is None:
                raise ValueError(f"Resim yüklenemedi: {image_path}")
            
            # Analiz başlat
            analysis_result = {
                "metadata": self._create_metadata(image_path),
                "color_analysis": self.color_analyzer.analyze(image),
                "form_size": self.shape_analyzer.analyze(image),
                "visual_features": self.feature_extractor.extract_features(image),
                "texture_patterns": self.texture_analyzer.analyze(image),
                "masking_used": should_mask and self.mask_processor is not None
            }
            
            # Güven skorları hesapla
            analysis_result["confidence_scores"] = self._calculate_confidence_scores(
                analysis_result
            )
            
            return analysis_result
            
        except Exception as e:
            raise RuntimeError(f"Analiz hatası: {str(e)}")
    
    def analyze_batch(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        """
        Birden fazla bagaj resmini toplu analiz eder.
        
        Args:
            image_paths: Analiz edilecek resimlerin yolları
            
        Returns:
            Her resim için analiz sonuçları listesi
        """
        results = []
        
        for image_path in image_paths:
            try:
                result = self.analyze_image(image_path)
                results.append(result)
            except Exception as e:
                # Hatalı resimleri atla, log tut
                error_result = {
                    "metadata": {"source_filename": image_path, "error": str(e)},
                    "status": "error"
                }
                results.append(error_result)
        
        return results
    
    def _create_metadata(self, image_path: str) -> Dict[str, Any]:
        """Analiz metadata'sını oluşturur."""
        return {
            "source_filename": image_path.split('/')[-1],
            "analysis_timestamp": datetime.now().isoformat(),
            "model_version": self.version,
            "config_hash": hash(str(self.config))
        }
    
    def _calculate_confidence_scores(self, analysis_result: Dict[str, Any]) -> Dict[str, float]:
        """Analiz sonuçları için güven skorları hesaplar."""
        scores = {}
        
        # Renk güven skoru
        color_data = analysis_result.get("color_analysis", {})
        if color_data.get("dominant_colors"):
            scores["color_confidence"] = min(1.0, len(color_data["dominant_colors"]) / 3.0)
        
        # Şekil güven skoru
        form_data = analysis_result.get("form_size", {})
        if form_data.get("suitcase_type") != "unknown":
            scores["shape_confidence"] = 0.8
        
        # Görsel özellik güven skoru
        visual_data = analysis_result.get("visual_features", {})
        if visual_data.get("sift_keypoints"):
            scores["feature_confidence"] = min(1.0, len(visual_data["sift_keypoints"]) / 50.0)
        
        # Genel güven skoru
        if scores:
            scores["overall_confidence"] = np.mean(list(scores.values()))
        
        return scores
    
    def _get_or_create_masked_image(self, image_path: str) -> Optional[str]:
        """
        Maskelenmiş resmi al veya oluştur
        
        Args:
            image_path: Orijinal resim yolu
            
        Returns:
            Maskelenmiş resim yolu veya None
        """
        if not self.mask_processor:
            return None
            
        try:
            from pathlib import Path
            
            # Orijinal dosya bilgilerini al
            original_path = Path(image_path)
            
            # Maskelenmiş dosya yolunu oluştur
            masked_filename = f"{original_path.stem}_masked{original_path.suffix}"
            
            # Config'den output dizinini al
            output_dir = self.config.get('processing', {}).get('output_dir', 'data/input-masked')
            masked_path = Path(output_dir) / masked_filename
            
            # Maskelenmiş dosya varsa kullan
            if masked_path.exists():
                return str(masked_path)
            
            # Yoksa oluştur
            result = self.mask_processor.process_single_image(
                image_path=original_path,
                overwrite=False,
                save_mask=True
            )
            
            if result["success"]:
                return result["output_file"]
            else:
                print(f"Maskeleme başarısız: {result.get('error', 'Bilinmeyen hata')}")
                return None
                
        except Exception as e:
            print(f"Maskeleme hatası: {e}")
            return None
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Varsayılan konfigürasyon değerleri."""
        return {
            "analysis": {
                "mode": "detailed",
                "features": {
                    "sift_features": 50,
                    "color_clusters": 5,
                    "texture_features": True,
                    "shape_analysis": True
                }
            },
            "masking": {
                "enabled": True,
                "sam_config": {
                    "model_type": "vit_h",
                    "device": "auto"
                }
            },
            "processing": {
                "output_dir": "data/input-masked"
            }
        }
