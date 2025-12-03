#!/usr/bin/env python3
"""
Pipeline Principal: Detecție Obiecte în Video 360° cu YOLO11
Integrează toate metodele de preprocessing și YOLO11.

Autor: Vladuceanu Tudor
"""

import sys
sys.path.append('.')

import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Tuple
import time
import logging
from dataclasses import dataclass, asdict
import json

# Preprocessing methods
from preprocessing.vertical_slice import VerticalSlicer
from preprocessing.cubemap import CubemapProjector

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configurație experiment"""
    method: str  # 'vertical_slice', 'cubemap'
    model_name: str  # 'yolo11n.pt', 'yolo11s.pt', etc.
    input_image: str
    output_dir: str
    
    # Vertical slice params
    num_slices: int = 6
    overlap_ratio: float = 0.15
    
    # Cubemap params
    face_size: int = 640
    
    # YOLO params
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.5


class Detection360Pipeline:
    """
    Pipeline complet pentru detecție în imagini 360°.
    """
    
    def __init__(self, config: ExperimentConfig):
        """
        Args:
            config: Configurație experiment
        """
        self.config = config
        self.results = {
            'config': asdict(config),
            'metrics': {},
            'detections': []
        }
        
        # Inițializează preprocessing method
        if config.method == 'vertical_slice':
            self.preprocessor = VerticalSlicer(
                num_slices=config.num_slices,
                overlap_ratio=config.overlap_ratio
            )
        elif config.method == 'cubemap':
            self.preprocessor = CubemapProjector(
                face_size=config.face_size
            )
        else:
            raise ValueError(f"Metodă necunoscută: {config.method}")
        
        # Inițializează YOLO model
        self.model = None
        self._load_model()
        
        logger.info(f"Pipeline inițializat: method={config.method}, "
                   f"model={config.model_name}")
    
    def _load_model(self):
        """Încarcă modelul YOLO11"""
        try:
            from ultralytics import YOLO
            
            model_path = Path('models') / self.config.model_name
            
            if not model_path.exists():
                # Download automat dacă nu există
                logger.info(f"Downloading {self.config.model_name}...")
                self.model = YOLO(self.config.model_name)
            else:
                self.model = YOLO(str(model_path))
            
            logger.info(f"✅ Model YOLO încărcat: {self.config.model_name}")
            
        except ImportError:
            logger.error("❌ Ultralytics nu este instalat! "
                        "Run: pip install ultralytics")
            raise
        except Exception as e:
            logger.error(f"❌ Eroare la încărcarea modelului: {e}")
            raise
    
    def process_image(self, image_path: str) -> Dict:
        """
        Procesează o imagine 360° completă.
        
        Args:
            image_path: Path la imagine equirectangular
            
        Returns:
            Dict cu rezultate:
                - detections: listă de detecții finale
                - metrics: metrici de performanță
                - visualizations: imagini cu rezultate
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Nu s-a putut citi imaginea: {image_path}")
        
        logger.info(f"Procesare imagine: {image.shape}")
        
        start_time = time.time()
        
        # Step 1: Preprocess
        tiles = self._preprocess(image)
        preprocess_time = time.time() - start_time
        
        # Step 2: Detect pe fiecare tile
        detect_start = time.time()
        all_detections = []
        
        for i, tile_info in enumerate(tiles):
            tile_dets = self._detect_on_tile(tile_info)
            all_detections.append(tile_dets)
            logger.debug(f"Tile {i}: {len(tile_dets)} detecții")
        
        detect_time = time.time() - detect_start
        
        # Step 3: Merge detections
        merge_start = time.time()
        final_detections = self._merge_detections(all_detections, tiles)
        merge_time = time.time() - merge_start
        
        total_time = time.time() - start_time
        
        # Compute metrics
        metrics = {
            'total_time': total_time,
            'preprocess_time': preprocess_time,
            'detect_time': detect_time,
            'merge_time': merge_time,
            'fps': 1.0 / total_time if total_time > 0 else 0,
            'num_tiles': len(tiles),
            'num_detections': len(final_detections),
            'avg_time_per_tile': detect_time / len(tiles) if tiles else 0
        }
        
        logger.info(f"✅ Procesare completă: {len(final_detections)} detecții "
                   f"în {total_time:.2f}s ({metrics['fps']:.2f} FPS)")
        
        # Visualize
        vis_image = self._visualize_detections(image, final_detections)
        
        return {
            'detections': final_detections,
            'metrics': metrics,
            'visualization': vis_image
        }
    
    def _preprocess(self, image: np.ndarray) -> List[Dict]:
        """Preprocess imagine în tiles"""
        if self.config.method == 'vertical_slice':
            return self.preprocessor.slice_image(image)
        elif self.config.method == 'cubemap':
            cubemap = self.preprocessor.equirectangular_to_cubemap(image)
            # Convert dict la list format pentru processing uniform
            return [
                {**face_data, 'face_name': face_name}
                for face_name, face_data in cubemap.items()
            ]
    
    def _detect_on_tile(self, tile_info: Dict) -> List[Dict]:
        """Rulează YOLO pe un tile"""
        tile_image = tile_info['image']
        
        # Run YOLO
        results = self.model(
            tile_image,
            conf=self.config.confidence_threshold,
            verbose=False
        )[0]
        
        # Parse rezultate
        detections = []
        
        if results.boxes is not None and len(results.boxes) > 0:
            boxes = results.boxes.xyxy.cpu().numpy()
            confidences = results.boxes.conf.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy().astype(int)
            
            for box, conf, cls in zip(boxes, confidences, classes):
                detections.append({
                    'bbox': box.tolist(),
                    'confidence': float(conf),
                    'class': int(cls),
                    'class_name': self.model.names[cls]
                })
        
        return detections
    
    def _merge_detections(
        self,
        all_detections: List[List[Dict]],
        tiles: List[Dict]
    ) -> List[Dict]:
        """Merge detecții de pe toate tiles"""
        # Map detecții la coordonate originale
        mapped_detections = []
        
        for tile_dets, tile_info in zip(all_detections, tiles):
            if self.config.method == 'vertical_slice':
                mapped = self.preprocessor.map_detections_to_original(
                    tile_dets, tile_info
                )
            elif self.config.method == 'cubemap':
                mapped = self.preprocessor.map_detections_to_equirectangular(
                    tile_dets, tile_info['metadata']
                )
            
            mapped_detections.append(mapped)
        
        # Merge cu NMS
        if self.config.method == 'vertical_slice':
            final = self.preprocessor.merge_overlapping_detections(
                mapped_detections,
                self.config.iou_threshold
            )
        elif self.config.method == 'cubemap':
            # Convert la dict format pentru cubemap
            dets_by_face = {}
            for dets in mapped_detections:
                for det in dets:
                    face = det.get('face_name', 'unknown')
                    if face not in dets_by_face:
                        dets_by_face[face] = []
                    dets_by_face[face].append(det)
            
            final = self.preprocessor.merge_cubemap_detections(
                dets_by_face,
                self.config.iou_threshold
            )
        
        return final
    
    def _visualize_detections(
        self,
        image: np.ndarray,
        detections: List[Dict]
    ) -> np.ndarray:
        """Desenează detecții pe imagine"""
        vis_image = image.copy()
        
        for det in detections:
            bbox = det['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            
            # Color per clasă
            class_id = det['class']
            color = self._get_color(class_id)
            
            # Draw box
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
            
            # Label
            label = f"{det['class_name']} {det['confidence']:.2f}"
            
            # Background pentru text
            (text_w, text_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(
                vis_image,
                (x1, y1 - text_h - 4),
                (x1 + text_w, y1),
                color,
                -1
            )
            
            # Text
            cv2.putText(
                vis_image,
                label,
                (x1, y1 - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
        
        return vis_image
    
    def _get_color(self, class_id: int) -> Tuple[int, int, int]:
        """Generează culoare pentru o clasă"""
        np.random.seed(class_id)
        return tuple(map(int, np.random.randint(0, 255, 3)))
    
    def save_results(self, output_dir: str = None):
        """Salvează rezultatele experimentului"""
        if output_dir is None:
            output_dir = self.config.output_dir
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Salvează metrics JSON
        metrics_file = output_path / 'metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"✅ Rezultate salvate în {output_dir}")


def main():
    """Main entry point pentru testare"""
    # Configurație experiment
    config = ExperimentConfig(
        method='vertical_slice',  # sau 'cubemap'
        model_name='yolo11n.pt',
        input_image='data/samples/test_360.jpg',
        output_dir='results/experiment_1',
        num_slices=6,
        overlap_ratio=0.15,
        confidence_threshold=0.25,
        iou_threshold=0.5
    )
    
    # Crează pipeline
    pipeline = Detection360Pipeline(config)
    
    # Procesează imagine
    results = pipeline.process_image(config.input_image)
    
    # Salvează rezultate
    output_path = Path(config.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    cv2.imwrite(
        str(output_path / 'visualization.jpg'),
        results['visualization']
    )
    
    # Print metrics
    print("\n" + "="*50)
    print("REZULTATE EXPERIMENT")
    print("="*50)
    print(f"Metodă: {config.method}")
    print(f"Model: {config.model_name}")
    print(f"Detecții: {results['metrics']['num_detections']}")
    print(f"Timp total: {results['metrics']['total_time']:.3f}s")
    print(f"FPS: {results['metrics']['fps']:.2f}")
    print("="*50)


if __name__ == '__main__':
    main()
