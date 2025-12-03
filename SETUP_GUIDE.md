# 🔬 Ghid Complet: Cercetare Detecție 360°

## 📦 Setup Inițial

### 1. Instalare Dependențe

```bash
# Creează environment virtual (recomandat)
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# sau
venv\Scripts\activate  # Windows

# Instalează dependențe
pip install --upgrade pip
pip install ultralytics opencv-python numpy pandas matplotlib seaborn tqdm
```

**Dependențe principale:**
- `ultralytics` - YOLO11 (include PyTorch)
- `opencv-python` - Procesare imagini
- `numpy` - Operații numerice
- `pandas` - Analiză date
- `matplotlib` + `seaborn` - Vizualizări
- `tqdm` - Progress bars

### 2. Structură Directoare

```bash
mkdir -p data/samples data/360_datasets
mkdir -p models
mkdir -p results
```

### 3. Download Date 360°

#### Opțiunea A: Dataset-uri Publice

**SUN360** (recomand pentru început):
```bash
# 1. Vizitează: http://people.csail.mit.edu/jxiao/SUN360/
# 2. Download imagini (selectează un subset mic ~100 imagini)
# 3. Extrage în data/360_datasets/sun360/
```

**360VOT** (pentru video tracking):
```bash
# 1. Vizitează: http://www.votchallenge.net/vot2021/dataset.html
# 2. Download 360° sequences
# 3. Extrage în data/360_datasets/360vot/
```

**Pano3D** (driving scenes):
```bash
# Vizitează: https://github.com/TRI-ML/packnet-sfm
# Follow instructions pentru download
```

#### Opțiunea B: Date Proprii

Dacă ai video-uri 360° proprii:
```bash
# Extract frame-uri din video
ffmpeg -i your_360_video.mp4 -vf fps=1 data/samples/frame_%04d.jpg
```

#### Opțiunea C: Date Sintetice (pentru testare rapidă)

Scriptul `benchmark.py` generează automat imagini dummy dacă nu există date.

### 4. Download Modele YOLO11

Modelele se descarcă automat la prima rulare, dar poți:

```bash
# Download manual (opțional)
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolo11n.pt -P models/
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolo11s.pt -P models/
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolo11m.pt -P models/
```

---

## 🚀 Quick Start

### Test Rapid cu Imagine Singură

```python
python detection_pipeline.py
```

Acest script va:
1. Descărca YOLO11n automat
2. Crea o imagine test dacă nu există
3. Rula detecția cu metoda `vertical_slice`
4. Salva rezultate în `results/experiment_1/`

### Benchmark Complet

```python
python benchmark.py
```

Aceasta va:
1. Testa toate metodele (vertical_slice + cubemap)
2. Cu YOLO11n (rapid pentru testare)
3. Genera plots comparative
4. Salva rapoarte în `results/benchmark/`

---

## 📊 Rulare Experimente Custom

### Experiment 1: Comparație Vertical Slices

```python
from detection_pipeline import Detection360Pipeline, ExperimentConfig

# Test cu 4 slices
config_4 = ExperimentConfig(
    method='vertical_slice',
    model_name='yolo11n.pt',
    input_image='data/samples/test.jpg',
    output_dir='results/slices_4',
    num_slices=4,
    overlap_ratio=0.15
)

pipeline_4 = Detection360Pipeline(config_4)
results_4 = pipeline_4.process_image(config_4.input_image)

# Test cu 8 slices
config_8 = ExperimentConfig(
    method='vertical_slice',
    model_name='yolo11n.pt',
    input_image='data/samples/test.jpg',
    output_dir='results/slices_8',
    num_slices=8,
    overlap_ratio=0.15
)

pipeline_8 = Detection360Pipeline(config_8)
results_8 = pipeline_8.process_image(config_8.input_image)

# Compară
print(f"4 slices: {results_4['metrics']['fps']:.2f} FPS")
print(f"8 slices: {results_8['metrics']['fps']:.2f} FPS")
```

### Experiment 2: Cubemap vs Vertical Slice

```python
# Vertical Slice
config_vs = ExperimentConfig(
    method='vertical_slice',
    model_name='yolo11s.pt',
    input_image='data/samples/test.jpg',
    output_dir='results/vs_test',
    num_slices=6
)

# Cubemap
config_cm = ExperimentConfig(
    method='cubemap',
    model_name='yolo11s.pt',
    input_image='data/samples/test.jpg',
    output_dir='results/cm_test',
    face_size=640
)

# Rulează ambele
pipeline_vs = Detection360Pipeline(config_vs)
pipeline_cm = Detection360Pipeline(config_cm)

results_vs = pipeline_vs.process_image(config_vs.input_image)
results_cm = pipeline_cm.process_image(config_cm.input_image)

# Compară acuratețe (dacă ai ground truth)
print(f"Vertical Slice: {len(results_vs['detections'])} detecții")
print(f"Cubemap: {len(results_cm['detections'])} detecții")
```

### Experiment 3: Toate Modelele YOLO

```python
models = ['yolo11n.pt', 'yolo11s.pt', 'yolo11m.pt']

for model in models:
    config = ExperimentConfig(
        method='vertical_slice',
        model_name=model,
        input_image='data/samples/test.jpg',
        output_dir=f'results/model_{model}',
        num_slices=6
    )
    
    pipeline = Detection360Pipeline(config)
    results = pipeline.process_image(config.input_image)
    
    print(f"{model}: {results['metrics']['fps']:.2f} FPS, "
          f"{len(results['detections'])} detecții")
```

---

## 📈 Analiză Rezultate

### 1. Citire Rezultate Benchmark

```python
import pandas as pd
import json

# CSV
df = pd.read_csv('results/benchmark/results.csv')
print(df.describe())

# JSON
with open('results/benchmark/results.json') as f:
    results = json.load(f)
```

### 2. Filtrare și Comparație

```python
# Best FPS per metodă
best_fps = df.groupby('method')['fps'].max()
print("Best FPS per method:")
print(best_fps)

# Best accuracy (num detections)
best_det = df.groupby('method')['num_detections'].mean()
print("\nAverage detections per method:")
print(best_det)

# Trade-off FPS vs Detections
import matplotlib.pyplot as plt

plt.scatter(df['fps'], df['num_detections'], 
           c=df['method'].astype('category').cat.codes)
plt.xlabel('FPS')
plt.ylabel('Number of Detections')
plt.title('FPS vs Detection Count')
plt.show()
```

### 3. Statistici Detaliate

```python
# Per metodă și model
summary = df.groupby(['method', 'model']).agg({
    'fps': ['mean', 'std'],
    'total_time': ['mean', 'std'],
    'num_detections': ['mean', 'std']
}).round(3)

print(summary)
```

---

## 🎯 Interpretarea Rezultatelor

### Ce să Căutăm:

**1. FPS (Frames Per Second)**
- **Înalt (>10 FPS)**: Bun pentru real-time pe Jetson
- **Mediu (5-10 FPS)**: Acceptabil pentru multe aplicații
- **Scăzut (<5 FPS)**: Probleme pentru deployment

**2. Număr Detecții**
- Comparați cu ground truth dacă există
- Prea puține = missed detections
- Prea multe = false positives

**3. Trade-offs**
- **Vertical Slice**: Mai rapid, dar distorsiuni polare
- **Cubemap**: Mai acurat, dar mai lent (6 inferențe)

### Decizii:

**Pentru Jetson AGX Xavier:**
- Dacă FPS > 15: Excelent pentru real-time
- Dacă FPS 10-15: Bun cu optimizări DeepStream
- Dacă FPS < 10: Consideră model mai mic (nano) sau mai puține tiles

**Recomandare:**
1. Începe cu `vertical_slice` + `yolo11n` + `6 slices`
2. Dacă acuratețea nu e suficientă → `yolo11s` sau `cubemap`
3. Dacă FPS-ul e prea mic → reduce num_slices sau folosește nano

---

## 🔧 Optimizări

### Pentru FPS Mai Mare:

**1. Reduce numărul de tiles:**
```python
config.num_slices = 4  # în loc de 6-8
```

**2. Folosește model mai mic:**
```python
config.model_name = 'yolo11n.pt'  # nano
```

**3. Increase confidence threshold:**
```python
config.confidence_threshold = 0.5  # mai puține false positives
```

**4. Batch processing (pentru cubemap):**
```python
# În loc să rulezi 6 inferențe separate,
# stack fețele cubului și rulează batch inference
# (necesită modificări în cod)
```

### Pentru Acuratețe Mai Mare:

**1. Increase overlap:**
```python
config.overlap_ratio = 0.25  # mai mult overlap între tiles
```

**2. Mai multe tiles:**
```python
config.num_slices = 8  # acoperire mai fină
```

**3. Model mai mare:**
```python
config.model_name = 'yolo11m.pt'  # medium
```

---

## 📝 Adnotare Date

Dacă ai imagini 360° fără adnotări:

### Folosind CVAT

```bash
# 1. Instalare CVAT (Docker)
git clone https://github.com/opencv/cvat
cd cvat
docker-compose up -d

# 2. Acces: http://localhost:8080
# 3. Upload imagini 360°
# 4. Crează task cu 360° mode
# 5. Adnotează obiecte
# 6. Export YOLO format
```

### Folosind Label Studio

```bash
pip install label-studio
label-studio start

# Acces: http://localhost:8080
# Import imagini și adnotează
```

---

## 🚀 Next Steps: Port la DeepStream

După ce ai metodă optimă:

**1. Exportă model pentru TensorRT:**
```python
from ultralytics import YOLO

model = YOLO('yolo11n.pt')
model.export(format='engine', imgsz=640)  # TensorRT
```

**2. Crează GStreamer pipeline:**
```python
# Pseudocod DeepStream config
[source]
type=uri
uri=file:///path/to/360_video.mp4

[preprocessing]
custom-lib=/path/to/libpreprocess360.so
num-slices=6

[primary-gie]
model-engine-file=yolo11n.engine
batch-size=6  # procesează toate tiles odată

[tracker]
# Optional tracking

[sink]
type=rtsp
```

**3. Implementează custom preprocessing plugin:**
- În C/C++ pentru DeepStream
- Folosește CUDA pentru preprocessing rapid
- Integrează logica de vertical_slice sau cubemap

---

## 📚 Resurse Suplimentare

**Papers:**
- "Distortion-Aware CNNs for Spherical Images" (IJCV 2019)
- "360-Indoor: Towards Learning Real-World Objects in 360° Indoor Equirectangular Images" (WACV 2020)
- "Kernel Transformer Networks for Compact Spherical Convolution" (CVPR 2019)

**Tools:**
- **py360convert**: Library pentru conversii 360°
- **equilib**: PyTorch library pentru equirectangular ops
- **Spherical-Package**: Rotații și transformări spherical

---

**Ready to start! 🚀**

Rulează `python benchmark.py` pentru primele rezultate!
