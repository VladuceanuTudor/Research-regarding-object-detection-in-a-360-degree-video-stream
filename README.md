# 🔬 Cercetare: Detecție Obiecte în Video 360°

## 📋 Obiectiv

Studiu comparativ al metodelor de pre-procesare a imaginilor 360° equirectangular pentru detecție obiecte cu YOLO, în vederea deployment pe Jetson AGX Xavier cu DeepStream.

## 🎯 Întrebări de Cercetare

1. **Care este metoda optimă de segmentare a frame-urilor 360°?**
   - Slicing vertical simple
   - Cubemap projection
   - Perspective projection (tangent images)
   - Fisheye unwrapping
   - ICO sphere projection

2. **Care este trade-off-ul între acuratețe și performanță?**
   - FPS per metodă
   - mAP (mean Average Precision)
   - Overlap handling între regiuni

3. **Cum gestionăm obiectele la granițele segmentelor?**
   - Overlap între tiles
   - Stitching inteligent
   - Non-maximum suppression global

## 📊 Metodologii de Testare

### Metodă 1: Vertical Slicing (Baseline)
```
Equirectangular → [Slice1|Slice2|Slice3|Slice4|Slice5|Slice6]
                   ↓       ↓       ↓       ↓       ↓       ↓
                  YOLO   YOLO   YOLO   YOLO   YOLO   YOLO
                   ↓       ↓       ↓       ↓       ↓       ↓
                  [Merge & NMS] → Final Detections
```

**Avantaje:**
- ✅ Simplu de implementat
- ✅ Rapid
- ✅ Funcționează cu orice model YOLO

**Dezavantaje:**
- ❌ Distorsiuni polare severe
- ❌ Obiecte tăiate la granițe
- ❌ Diferențe mari de scară

### Metodă 2: Cubemap Projection
```
Equirectangular → [Front][Back][Left][Right][Top][Bottom]
                   ↓      ↓     ↓     ↓      ↓     ↓
                  YOLO  YOLO  YOLO  YOLO  YOLO  YOLO
                   ↓      ↓     ↓     ↓      ↓     ↓
                  [Merge & Transform back] → Final
```

**Avantaje:**
- ✅ Fără distorsiuni în centrul fiecărei fețe
- ✅ Scale uniformă
- ✅ Acoperire completă

**Dezavantaje:**
- ❌ Mai complex
- ❌ 6 inferențe necesare
- ❌ Obiecte la muchii pot fi tăiate

### Metodă 3: Perspective Tangent Images
```
Equirectangular → [N][NE][E][SE][S][SW][W][NW]
                   (8 perspective views cu overlap)
```

**Avantaje:**
- ✅ Perspective naturale
- ✅ Bună pentru obiecte îndepărtate
- ✅ Overlap controlabil

**Dezavantaje:**
- ❌ 8+ inferențe
- ❌ Computație mare

### Metodă 4: Adaptive Grid (Hibrid)
```
Ecuator: slicing vertical fin (multe obiecte)
Poli: tiles mai mari (mai puține obiecte)
```

## 📁 Structura Proiect

```
360-detection-research/
├── data/
│   ├── 360_datasets/          # Dataset-uri 360° cu adnotări
│   ├── samples/               # Sample images pentru test rapid
│   └── annotations/           # Ground truth annotations
├── models/
│   ├── yolo11n.pt            # YOLO11 nano
│   ├── yolo11s.pt            # YOLO11 small
│   └── yolo11m.pt            # YOLO11 medium
├── preprocessing/
│   ├── vertical_slice.py     # Metodă 1
│   ├── cubemap.py            # Metodă 2
│   ├── tangent_images.py     # Metodă 3
│   └── adaptive_grid.py      # Metodă 4
├── evaluation/
│   ├── metrics.py            # mAP, FPS, etc.
│   ├── visualize.py          # Vizualizare rezultate
│   └── compare.py            # Comparație metode
└── results/
    └── experiments/          # Rezultate per metodă
```

## 🗄️ Surse de Date 360°

### Dataset-uri Publice:

1. **Stanford 2D-3D-S** (Indoor 360°)
   - URL: http://buildingparser.stanford.edu/dataset.html
   - Conține: RGB panoramas + depth + annotations
   - Obiecte: furniture, doors, windows

2. **Matterport3D** (Indoor 360°)
   - URL: https://niessner.github.io/Matterport/
   - Massive indoor dataset
   - Requires academic license

3. **SUN360** (Outdoor/Indoor mix)
   - URL: http://people.csail.mit.edu/jxiao/SUN360/
   - 360° panoramas diverse scenes

4. **Pano3D** (Outdoor driving)
   - Street-level 360° images
   - Good for vehicle/pedestrian detection

5. **360VOT** (Video Object Tracking 360°)
   - Video sequences with tracking annotations
   - Perfect pentru testare DeepStream

### Generare Date Sintetice:

- **CARLA Simulator** cu camera 360° custom
- **Unity** cu 360° camera rendering
- **Blender** cu equirectangular rendering

### Adnotare:

**Dacă ai video-uri 360° fără adnotări:**
- **CVAT** (Computer Vision Annotation Tool) - suportă 360°
- **Labelbox** - 360° annotation support
- **Label Studio** - custom 360° labeling

## 🚀 Pipeline Experimentare

```python
# Pseudocod workflow
for method in [vertical_slice, cubemap, tangent, adaptive]:
    for model in [yolo11n, yolo11s, yolo11m]:
        # 1. Preprocess
        tiles = method.split_360_frame(frame)
        
        # 2. Detect
        detections = []
        for tile in tiles:
            dets = model.predict(tile)
            detections.append(dets)
        
        # 3. Merge
        final_dets = merge_detections(detections, method)
        
        # 4. Evaluate
        metrics = evaluate(final_dets, ground_truth)
        
        # 5. Save results
        save_results(method, model, metrics)

# 6. Compare all methods
compare_and_visualize(all_results)
```

## 📊 Metrici de Evaluare

1. **Acuratețe:**
   - mAP@0.5
   - mAP@0.5:0.95
   - Per-class precision/recall

2. **Performanță:**
   - FPS (frames per second)
   - Latency (ms)
   - Memory usage

3. **Calitate Merge:**
   - Duplicate detections rate
   - Split objects rate
   - Boundary accuracy

## 🎯 Next Steps

1. ✅ Setup environment Python
2. ✅ Download YOLO11 models
3. ✅ Implementare metode preprocessing
4. ✅ Test pe sample images
5. ✅ Download sau generare dataset
6. ✅ Rulare experimente comprehensive
7. ✅ Analiză rezultate
8. ✅ Selectare metodă optimă
9. ✅ Port la DeepStream pe Jetson
