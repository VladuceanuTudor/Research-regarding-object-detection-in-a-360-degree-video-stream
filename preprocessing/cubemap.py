#!/usr/bin/env python3
"""
Metodă 2: Cubemap Projection pentru imagini 360°
Transformă equirectangular în 6 fețe cubice fără distorsiuni.

Autor: Vladuceanu Tudor
"""

import numpy as np
import cv2
from typing import List, Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CubemapProjector:
    """
    Proiectează imagini 360° equirectangular pe un cubemap (6 fețe).
    """
    
    # Orientări pentru cele 6 fețe ale cubului
    FACES = {
        'front':  {'yaw': 0,    'pitch': 0,    'roll': 0},
        'right':  {'yaw': 90,   'pitch': 0,    'roll': 0},
        'back':   {'yaw': 180,  'pitch': 0,    'roll': 0},
        'left':   {'yaw': -90,  'pitch': 0,    'roll': 0},
        'top':    {'yaw': 0,    'pitch': 90,   'roll': 0},
        'bottom': {'yaw': 0,    'pitch': -90,  'roll': 0},
    }
    
    def __init__(self, face_size: int = 512):
        """
        Args:
            face_size: Dimensiunea fiecărei fețe cubice (px)
        """
        self.face_size = face_size
        logger.info(f"CubemapProjector inițializat: face_size={face_size}")
        
    def equirectangular_to_cubemap(
        self, 
        equirect_image: np.ndarray
    ) -> Dict[str, Dict]:
        """
        Convertește imagine equirectangular în cubemap (6 fețe).
        
        Args:
            equirect_image: Imagine panoramică (H, W, 3)
            
        Returns:
            Dict cu 6 fețe:
                {'front': {'image': np.ndarray, 'metadata': dict}, ...}
        """
        h, w = equirect_image.shape[:2]
        
        cubemap_faces = {}
        
        for face_name, angles in self.FACES.items():
            face_image = self._generate_face(
                equirect_image,
                angles['yaw'],
                angles['pitch'],
                angles['roll']
            )
            
            cubemap_faces[face_name] = {
                'image': face_image,
                'metadata': {
                    'face_name': face_name,
                    'yaw': angles['yaw'],
                    'pitch': angles['pitch'],
                    'roll': angles['roll'],
                    'face_size': self.face_size,
                    'original_shape': (h, w)
                }
            }
        
        logger.debug(f"Generat cubemap cu {len(cubemap_faces)} fețe")
        return cubemap_faces
    
    def _generate_face(
        self,
        equirect: np.ndarray,
        yaw: float,
        pitch: float,
        roll: float
    ) -> np.ndarray:
        """
        Generează o față a cubului din imaginea equirectangular.
        
        Uses perspective projection cu FOV de 90° pentru fiecare față.
        """
        h_eq, w_eq = equirect.shape[:2]
        
        # Mesh grid pentru fața cubului
        x = np.linspace(-1, 1, self.face_size)
        y = np.linspace(-1, 1, self.face_size)
        xx, yy = np.meshgrid(x, y)
        
        # Proiecție perspective → sphere coordinates
        zz = np.ones_like(xx)
        
        # Rotație pentru orientarea fiecărei fețe
        yaw_rad = np.deg2rad(yaw)
        pitch_rad = np.deg2rad(pitch)
        
        # Aplicăre rotații (simplified rotation matrices)
        # În practică, pentru producție ar trebui să folosim rotații 3D complete
        
        # Convert to spherical coordinates
        r = np.sqrt(xx**2 + yy**2 + zz**2)
        theta = np.arctan2(xx, zz) + yaw_rad  # azimuth
        phi = np.arcsin(yy / r) + pitch_rad    # elevation
        
        # Spherical to equirectangular mapping
        u = (theta / (2 * np.pi) + 0.5) * w_eq
        v = (0.5 - phi / np.pi) * h_eq
        
        # Clamp coordinates
        u = np.clip(u, 0, w_eq - 1).astype(np.float32)
        v = np.clip(v, 0, h_eq - 1).astype(np.float32)
        
        # Remap folosind cv2
        face_image = cv2.remap(
            equirect,
            u, v,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_WRAP
        )
        
        return face_image
    
    def map_detections_to_equirectangular(
        self,
        detections: List[Dict],
        face_metadata: Dict
    ) -> List[Dict]:
        """
        Mapează detecțiile dintr-o față cubemap înapoi la equirectangular.
        
        Args:
            detections: Listă de detecții în coordonate față cubică
            face_metadata: Metadata din equirectangular_to_cubemap()
            
        Returns:
            Detecții în coordonate equirectangular
        """
        yaw = face_metadata['yaw']
        pitch = face_metadata['pitch']
        face_size = face_metadata['face_size']
        h_orig, w_orig = face_metadata['original_shape']
        
        yaw_rad = np.deg2rad(yaw)
        pitch_rad = np.deg2rad(pitch)
        
        mapped_detections = []
        
        for det in detections:
            bbox = det['bbox']  # [x1, y1, x2, y2] în coordonate față
            
            # Convert bbox corners la coordonate equirectangular
            corners = [
                (bbox[0], bbox[1]),  # top-left
                (bbox[2], bbox[1]),  # top-right
                (bbox[2], bbox[3]),  # bottom-right
                (bbox[0], bbox[3]),  # bottom-left
            ]
            
            eq_corners = []
            for x_face, y_face in corners:
                # Normalize la [-1, 1]
                x_norm = (x_face / face_size) * 2 - 1
                y_norm = (y_face / face_size) * 2 - 1
                
                # 3D point pe fața cubului
                z_norm = 1.0
                
                # Rotație inversă
                theta = np.arctan2(x_norm, z_norm) + yaw_rad
                r = np.sqrt(x_norm**2 + y_norm**2 + z_norm**2)
                phi = np.arcsin(y_norm / r) + pitch_rad
                
                # Map la equirectangular
                u_eq = (theta / (2 * np.pi) + 0.5) * w_orig
                v_eq = (0.5 - phi / np.pi) * h_orig
                
                eq_corners.append((u_eq, v_eq))
            
            # Bounding box în equirectangular (axis-aligned)
            eq_xs = [c[0] for c in eq_corners]
            eq_ys = [c[1] for c in eq_corners]
            
            x1_eq = int(max(0, min(eq_xs)))
            y1_eq = int(max(0, min(eq_ys)))
            x2_eq = int(min(w_orig - 1, max(eq_xs)))
            y2_eq = int(min(h_orig - 1, max(eq_ys)))
            
            mapped_det = det.copy()
            mapped_det['bbox'] = [x1_eq, y1_eq, x2_eq, y2_eq]
            mapped_det['face_name'] = face_metadata['face_name']
            mapped_detections.append(mapped_det)
        
        return mapped_detections
    
    def merge_cubemap_detections(
        self,
        all_detections: Dict[str, List[Dict]],
        iou_threshold: float = 0.5
    ) -> List[Dict]:
        """
        Merge detecții din toate cele 6 fețe.
        
        Args:
            all_detections: Dict {'face_name': [detecții]}
            iou_threshold: Threshold pentru NMS
            
        Returns:
            Listă finală de detecții
        """
        # Flatten toate detecțiile
        all_dets = []
        for face_name, dets in all_detections.items():
            all_dets.extend(dets)
        
        if not all_dets:
            return []
        
        # Group by class și aplică NMS
        detections_by_class = {}
        for det in all_dets:
            cls = det['class']
            if cls not in detections_by_class:
                detections_by_class[cls] = []
            detections_by_class[cls].append(det)
        
        final_detections = []
        for cls, dets in detections_by_class.items():
            nms_dets = self._nms(dets, iou_threshold)
            final_detections.extend(nms_dets)
        
        logger.info(f"Cubemap merge: {len(all_dets)} detecții → {len(final_detections)} finale")
        return final_detections
    
    def _nms(self, detections: List[Dict], iou_threshold: float) -> List[Dict]:
        """Non-Maximum Suppression"""
        if not detections:
            return []
        
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
        """Intersection over Union"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def visualize_cubemap(
        self,
        cubemap_faces: Dict[str, Dict],
        output_path: str = None
    ) -> np.ndarray:
        """
        Vizualizează cubemap-ul ca un grid 3x4.
        
        Layout:
            [  ] [Top] [  ] [  ]
            [Left][Front][Right][Back]
            [  ] [Bottom] [  ] [  ]
        """
        face_size = self.face_size
        
        # Create blank canvas
        canvas = np.zeros((face_size * 3, face_size * 4, 3), dtype=np.uint8)
        
        # Place faces
        placements = {
            'top':    (0, 1),
            'left':   (1, 0),
            'front':  (1, 1),
            'right':  (1, 2),
            'back':   (1, 3),
            'bottom': (2, 1),
        }
        
        for face_name, (row, col) in placements.items():
            y_start = row * face_size
            x_start = col * face_size
            canvas[y_start:y_start+face_size, x_start:x_start+face_size] = \
                cubemap_faces[face_name]['image']
        
        if output_path:
            cv2.imwrite(output_path, canvas)
            logger.info(f"Cubemap salvat în {output_path}")
        
        return canvas


def demo():
    """Demonstrație funcționare"""
    # Crează imagine equirectangular simulată
    img = np.random.randint(0, 255, (1024, 2048, 3), dtype=np.uint8)
    
    # Draw some patterns pentru a vedea transformarea
    cv2.circle(img, (1024, 512), 100, (0, 255, 0), -1)  # centru
    cv2.rectangle(img, (100, 100), (300, 300), (255, 0, 0), 5)  # stânga
    
    # Inițializează projector
    projector = CubemapProjector(face_size=512)
    
    # Generate cubemap
    cubemap = projector.equirectangular_to_cubemap(img)
    
    print(f"\n✅ Generat cubemap cu {len(cubemap)} fețe:")
    for face_name, face_data in cubemap.items():
        print(f"  {face_name}: shape={face_data['image'].shape}")
    
    # Vizualizează
    grid = projector.visualize_cubemap(cubemap, '/tmp/cubemap_demo.jpg')
    print(f"\n✅ Cubemap grid salvat: {grid.shape}")


if __name__ == '__main__':
    demo()
