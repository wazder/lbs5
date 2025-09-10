"""
Test dosyası - Maskeleme sistemi
"""

import unittest
import numpy as np
import cv2
import tempfile
import os
from pathlib import Path

from src.mask.sam_detector import SAMDetector
from src.mask.mask_processor import MaskProcessor

class TestMaskingSystem(unittest.TestCase):
    """
    Maskeleme sistemi test sınıfı
    """
    
    def setUp(self):
        """
        Test ortamını hazırla
        """
        # Test resmi oluştur
        self.test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Geçici klasörler
        self.temp_dir = tempfile.mkdtemp()
        self.input_dir = Path(self.temp_dir) / "input"
        self.output_dir = Path(self.temp_dir) / "output"
        
        self.input_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        
        # Test resmi kaydet
        self.test_image_path = self.input_dir / "test.jpg"
        cv2.imwrite(str(self.test_image_path), self.test_image)
    
    def tearDown(self):
        """
        Test sonrası temizlik
        """
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_sam_detector_initialization(self):
        """
        SAM dedektör başlatma testi
        """
        detector = SAMDetector(device="cpu")
        self.assertIsInstance(detector, SAMDetector)
    
    def test_mask_processor_initialization(self):
        """
        Mask processor başlatma testi
        """
        processor = MaskProcessor(
            input_dir=str(self.input_dir),
            output_dir=str(self.output_dir)
        )
        self.assertIsInstance(processor, MaskProcessor)
        self.assertTrue(processor.output_dir.exists())
    
    def test_fallback_detection(self):
        """
        Fallback tespit yöntemini test et
        """
        detector = SAMDetector(device="cpu")
        detector.sam_available = False  # SAM'i devre dışı bırak
        
        mask = detector.detect_luggage(self.test_image)
        self.assertIsNotNone(mask)
        self.assertEqual(mask.shape[:2], self.test_image.shape[:2])
    
    def test_statistics(self):
        """
        İstatistik fonksiyonunu test et
        """
        processor = MaskProcessor(
            input_dir=str(self.input_dir),
            output_dir=str(self.output_dir)
        )
        
        stats = processor.get_statistics()
        self.assertIn("input_files", stats)
        self.assertIn("output_files", stats)
        self.assertEqual(stats["input_files"], 1)  # test.jpg
    
    def test_load_image(self):
        """
        Resim yükleme testi
        """
        processor = MaskProcessor(
            input_dir=str(self.input_dir),
            output_dir=str(self.output_dir)
        )
        
        loaded_image = processor._load_image(self.test_image_path)
        self.assertIsNotNone(loaded_image)
        self.assertEqual(loaded_image.shape, self.test_image.shape)

if __name__ == "__main__":
    unittest.main()
