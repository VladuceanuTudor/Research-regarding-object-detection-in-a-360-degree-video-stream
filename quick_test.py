#!/usr/bin/env python3
"""
Test Rapid: Verifică că totul funcționează
Rulează un experiment simplu pentru validare setup.

Autor: Vladuceanu Tudor
"""

import sys
from pathlib import Path
import cv2
import numpy as np

print("="*60)
print("🧪 TEST RAPID - 360° Detection Pipeline")
print("="*60)
print()

# Test 1: Verifică imports
print("Test 1: Verificare imports...")
try:
    from ultralytics import YOLO
    print("  ✅ ultralytics (YOLO) - OK")
except ImportError as e:
    print(f"  ❌ ultralytics NU este instalat!")
    print(f"     Rulează: pip install ultralytics")
    sys.exit(1)

try:
    import pandas as pd
    print("  ✅ pandas - OK")
except ImportError:
    print("  ⚠️  pandas NU este instalat (pentru benchmark)")
    print("     Rulează: pip install pandas matplotlib seaborn")

print()

# Test 2: Verifică structură directoare
print("Test 2: Verificare directoare...")
dirs = ['data/samples', 'models', 'results', 'preprocessing']

for dir_path in dirs:
    path = Path(dir_path)
    if path.exists():
        print(f"  ✅ {dir_path} - OK")
    else:
        print(f"  ⚠️  {dir_path} - creez...")
        path.mkdir(parents=True, exist_ok=True)
print()

# Test 3: Crează imagine test dacă nu există
print("Test 3: Verificare imagini test...")
sample_path = Path('data/samples/test_360.jpg')

if not sample_path.exists():
    print("  ⏳ Creare imagine test 360°...")
    
    # Crează imagine equirectangular simulată
    test_img = np.random.randint(0, 255, (1024, 2048, 3), dtype=np.uint8)
    
    # Adaugă unele obiecte pentru testare
    # Simulează "obiecte" - pătrate colorate
    cv2.rectangle(test_img, (200, 300), (400, 500), (0, 255, 0), -1)
    cv2.rectangle(test_img, (800, 400), (1000, 600), (255, 0, 0), -1)
    cv2.rectangle(test_img, (1500, 200), (1700, 400), (0, 0, 255), -1)
    cv2.circle(test_img, (1024, 512), 150, (255, 255, 0), -1)
    
    cv2.imwrite(str(sample_path), test_img)
    print(f"  ✅ Imagine test creată: {sample_path}")
else:
    print(f"  ✅ Imagine test există: {sample_path}")
print()

# Test 4: Download model YOLO (dacă nu există)
print("Test 4: Verificare model YOLO11...")
try:
    model = YOLO('yolo11n.pt')
    print("  ✅ yolo11n.pt - încărcat")
except Exception as e:
    print(f"  ⚠️  Eroare la încărcare model: {e}")
    print("     Se va descărca automat la prima rulare")
print()

# Test 5: Test preprocessing
print("Test 5: Test preprocessing...")
try:
    from preprocessing.vertical_slice import VerticalSlicer
    
    slicer = VerticalSlicer(num_slices=4)
    test_img = np.random.randint(0, 255, (512, 1024, 3), dtype=np.uint8)
    slices = slicer.slice_image(test_img)
    
    print(f"  ✅ VerticalSlicer - OK ({len(slices)} slices creați)")
except Exception as e:
    print(f"  ❌ VerticalSlicer - EROARE: {e}")
print()

try:
    from preprocessing.cubemap import CubemapProjector
    
    projector = CubemapProjector(face_size=256)
    test_img = np.random.randint(0, 255, (512, 1024, 3), dtype=np.uint8)
    cubemap = projector.equirectangular_to_cubemap(test_img)
    
    print(f"  ✅ CubemapProjector - OK ({len(cubemap)} fețe create)")
except Exception as e:
    print(f"  ❌ CubemapProjector - EROARE: {e}")
print()

# Test 6: Test pipeline complet (dacă totul e OK)
print("Test 6: Test pipeline complet...")
try:
    sys.path.insert(0, '.')
    from detection_pipeline import Detection360Pipeline, ExperimentConfig
    
    config = ExperimentConfig(
        method='vertical_slice',
        model_name='yolo11n.pt',
        input_image=str(sample_path),
        output_dir='results/quick_test',
        num_slices=4,
        confidence_threshold=0.25
    )
    
    print("  ⏳ Rulare pipeline test...")
    pipeline = Detection360Pipeline(config)
    results = pipeline.process_image(str(sample_path))
    
    print(f"  ✅ Pipeline completat!")
    print(f"     - Timp procesare: {results['metrics']['total_time']:.2f}s")
    print(f"     - FPS: {results['metrics']['fps']:.2f}")
    print(f"     - Detecții: {len(results['detections'])}")
    print(f"     - Rezultate în: results/quick_test/")
    
except Exception as e:
    print(f"  ❌ Pipeline - EROARE: {e}")
    import traceback
    traceback.print_exc()
print()

# Summary
print("="*60)
print("📊 REZUMAT TEST")
print("="*60)
print()
print("✅ Setup complet! Poți rula:")
print()
print("  1. python detection_pipeline.py    # Test cu o imagine")
print("  2. python benchmark.py             # Benchmark complet")
print()
print("📁 Verifică rezultate în:")
print("  - results/quick_test/              # Acest test")
print("  - results/benchmark/               # După benchmark")
print()
print("="*60)
