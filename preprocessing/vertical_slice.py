#!/usr/bin/env python3
"""
Metodă 1: Vertical Slicing pentru imagini 360°
Cea mai simplă metodă - împarte imaginea equirectangular în benzi verticale.

Autor: Vladuceanu Tudor
"""

import numpy as np
import cv2
from typing import List, Tuple, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VerticalSlicer:
    """
    Segmentează imagini 360° equirectangular în benzi verticale.
    """
    
    def __init__(
        self, 
        num_slices: int = 6,
        overlap_ratio: float = 0.1,
        min_slice_width: int = 640
    ):
        """
        Args:
            num_slices: Numărul de benzi verticale
            overlap_ratio: Procentul de overlap între benzi (0.0-0.5)
            min_slice_width: Lățime minimă per slice (pentru resize)
        """
        self.num_slices = num_slices
        self.overlap_ratio = overlap_ratio
        self.min_slice_width = min_slice_width
        
        logger.info(f"VerticalSlicer inițializat: {num_slices} slices, "
                   f"overlap={overlap_ratio}, min_width={min_slice_width}")
    
    def slice_image(self, image: np.ndarray) -> List[Dict]:
        """
        Segmentează imaginea în slices verticale cu overlap.
        
        Args:
            image: Imagine equirectangular (H, W, 3)
            
        Returns:
            List de dict-uri cu:
                - 'image': numpy array al slice-ului
                - 'x_start': coordonata x de început în imaginea originală
                - 'x_end': coordonata x de final
                - 'width': lățimea slice-ului
                - 'original_shape': shape-ul imaginii originale
        """
        h, w = image.shape[:2]
        
        # Calculează lățimea fiecărui slice cu overlap
        overlap_pixels = int(w / self.num_slices * self.overlap_ratio)
        slice_width = w // self.num_slices + overlap_pixels
        
        slices = []
        
        for i in range(self.num_slices):
            # Coordonate de start/end cu overlap
            x_start = max(0, i * (w // self.num_slices) - overlap_pixels // 2)
            x_end = min(w, x_start + slice_width)
            
            # Handle wrap-around pentru ultimul slice (360° continuitate)
            if i == self.num_slices - 1 and x_end < w:
                # Extinde până la margine
                x_end = w
                # Adaugă și partea de la început pentru continuitate
                wrap_pixels = slice_width - (x_end - x_start)
                if wrap_pixels > 0:
                    slice_img = np.concatenate([
                        image[:, x_start:x_end],
                        image[:, 0:wrap_pixels]
                    ], axis=1)
                else:
                    slice_img = image[:, x_start:x_end]
            else:
                slice_img = image[:, x_start:x_end]
            
            # Resize dacă este prea mic
            if slice_img.shape[1] < self.min_slice_width:
                scale = self.min_slice_width / slice_img.shape[1]
                new_h = int(slice_img.shape[0] * scale)
                slice_img = cv2.resize(slice_img, (self.min_slice_width, new_h))
            
            slices.append({
                'image': slice_img,
                'x_start': x_start,
                'x_end': x_end,
                'width': x_end - x_start,
                'original_shape': (h, w),
                'slice_id': i,
                'wrap_around': i == self.num_slices - 1 and wrap_pixels > 0
            })
        
        logger.debug(f"Creat {len(slices)} slices din imagine {w}x{h}")
        return slices
    
    def map_detections_to_original(
        self, 
        detections: List[Dict], 
        slice_info: Dict
    ) -> List[Dict]:
        """
        Mapează detectările dintr-un slice înapoi la coordonatele originale.
        
        Args:
            detections: Lista de detecții format YOLO:
                [{'bbox': [x1, y1, x2, y2], 'confidence': float, 'class': int}]
            slice_info: Info despre slice din slice_image()
            
        Returns:
            Detecții cu coordonate în spațiul original
        """
        original_h, original_w = slice_info['original_shape']
        x_start = slice_info['x_start']
        slice_id = slice_info['slice_id']
        
        mapped_detections = []
        
        for det in detections:
            bbox = det['bbox']  # [x1, y1, x2, y2] în coordonate slice
            
            # Scale factor dacă slice-ul a fost resized
            scale_x = slice_info['width'] / slice_info['image'].shape[1]
            scale_y = original_h / slice_info['image'].shape[0]
            
            # Mapează la coordonate originale
            x1_orig = int(bbox[0] * scale_x + x_start)
            y1_orig = int(bbox[1] * scale_y)
            x2_orig = int(bbox[2] * scale_x + x_start)
            y2_orig = int(bbox[3] * scale_y)
            
            # Handle wrap-around
            if slice_info.get('wrap_around'):
                if x1_orig >= original_w:
                    x1_orig -= original_w
                if x2_orig >= original_w:
                    x2_orig -= original_w
            
            # Clamp la dimensiunile imaginii
            x1_orig = max(0, min(x1_orig, original_w - 1))
            x2_orig = max(0, min(x2_orig, original_w - 1))
            y1_orig = max(0, min(y1_orig, original_h - 1))
            y2_orig = max(0, min(y2_orig, original_h - 1))
            
            mapped_det = det.copy()
            mapped_det['bbox'] = [x1_orig, y1_orig, x2_orig, y2_orig]
            mapped_det['slice_id'] = slice_id
            mapped_detections.append(mapped_det)
        
        return mapped_detections
    
    def merge_overlapping_detections(
        self, 
        all_detections: List[List[Dict]],
        iou_threshold: float = 0.5
    ) -> List[Dict]:
        """
        Merge detecții din multiple slices folosind Non-Maximum Suppression.
        
        Args:
            all_detections: Lista de liste cu detecții per slice
            iou_threshold: Threshold pentru NMS
            
        Returns:
            Lista de detecții finale după merge
        """
        # Flatten toate detecțiile
        all_dets = []
        for slice_dets in all_detections:
            all_dets.extend(slice_dets)
        
        if not all_dets:
            return []
        
        # Grupează pe clase
        detections_by_class = {}
        for det in all_dets:
            cls = det['class']
            if cls not in detections_by_class:
                detections_by_class[cls] = []
            detections_by_class[cls].append(det)
        
        # Aplică NMS per clasă
        final_detections = []
        for cls, dets in detections_by_class.items():
            nms_dets = self._nms(dets, iou_threshold)
            final_detections.extend(nms_dets)
        
        logger.info(f"Merge: {len(all_dets)} detecții → {len(final_detections)} finale")
        return final_detections
    
    def _nms(self, detections: List[Dict], iou_threshold: float) -> List[Dict]:
        """Non-Maximum Suppression"""
        if not detections:
            return []
        
        # Sort by confidence
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        
        keep = []
        while detections:
            best = detections.pop(0)
            keep.append(best)
            
            detections = [
                det for det in detections 
                if self._iou(best['bbox'], det['bbox']) < iou_threshold
            ]
        
        return keep
    
    def _iou(self, box1: List[int], box2: List[int]) -> float:
        """Calculează Intersection over Union"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0


def demo():
    """Demonstrație funcționare"""
    # Crează imagine test 360° simulată
    img = np.random.randint(0, 255, (1024, 2048, 3), dtype=np.uint8)
    
    # Inițializează slicer
    slicer = VerticalSlicer(num_slices=6, overlap_ratio=0.15)
    
    # Slice imagine
    slices = slicer.slice_image(img)
    
    print(f"\n✅ Creat {len(slices)} slices:")
    for i, s in enumerate(slices):
        print(f"  Slice {i}: shape={s['image'].shape}, "
              f"x_range=[{s['x_start']}, {s['x_end']}]")
    
    # Simulează detecții
    fake_detections = [
        [{'bbox': [100, 100, 200, 200], 'confidence': 0.9, 'class': 0}]
        for _ in slices
    ]
    
    # Map înapoi la original
    all_mapped = []
    for dets, slice_info in zip(fake_detections, slices):
        mapped = slicer.map_detections_to_original(dets, slice_info)
        all_mapped.append(mapped)
    
    # Merge
    final = slicer.merge_overlapping_detections(all_mapped)
    print(f"\n✅ După merge: {len(final)} detecții finale")


if __name__ == '__main__':
    demo()
