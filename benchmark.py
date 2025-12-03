#!/usr/bin/env python3
"""
Benchmark Complet: Comparație Metode de Preprocessing pentru Detecție 360°

Rulează experimente cu toate combinațiile de:
- Metode preprocessing (vertical_slice, cubemap)
- Modele YOLO (yolo11n, yolo11s, yolo11m)

Autor: Vladuceanu Tudor
"""

import sys
sys.path.append('.')

from pathlib import Path
import json
import time
from typing import List, Dict
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import logging

from detection_pipeline import Detection360Pipeline, ExperimentConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Seaborn style pentru plot-uri
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


class Benchmark360:
    """
    Sistemul de benchmark pentru evaluarea metodelor 360°.
    """
    
    def __init__(self, output_dir: str = 'results/benchmark'):
        """
        Args:
            output_dir: Director pentru rezultate
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results = []
        
        logger.info(f"Benchmark inițializat: output_dir={output_dir}")
    
    def run_experiments(
        self,
        test_images: List[str],
        methods: List[str] = ['vertical_slice', 'cubemap'],
        models: List[str] = ['yolo11n.pt', 'yolo11s.pt'],
        num_slices_variants: List[int] = [4, 6, 8],
        face_sizes: List[int] = [512, 640]
    ):
        """
        Rulează experimente complete.
        
        Args:
            test_images: Listă de path-uri la imagini test
            methods: Metode de preprocessing de testat
            models: Modele YOLO de testat
            num_slices_variants: Variante pentru num_slices (vertical_slice)
            face_sizes: Variante pentru face_size (cubemap)
        """
        total_experiments = (
            len(test_images) * 
            (len(methods) * len(models) * 
             max(len(num_slices_variants), len(face_sizes)))
        )
        
        logger.info(f"🚀 Pornire benchmark: {total_experiments} experimente")
        
        pbar = tqdm(total=total_experiments, desc="Running experiments")
        
        for img_idx, image_path in enumerate(test_images):
            image_name = Path(image_path).stem
            
            for method in methods:
                for model in models:
                    
                    if method == 'vertical_slice':
                        variants = num_slices_variants
                        param_name = 'num_slices'
                    else:  # cubemap
                        variants = face_sizes
                        param_name = 'face_size'
                    
                    for variant in variants:
                        # Configurație experiment
                        config = ExperimentConfig(
                            method=method,
                            model_name=model,
                            input_image=image_path,
                            output_dir=str(self.output_dir / 
                                         f"{image_name}_{method}_{model}_{param_name}{variant}"),
                            num_slices=variant if method == 'vertical_slice' else 6,
                            face_size=variant if method == 'cubemap' else 640,
                            overlap_ratio=0.15,
                            confidence_threshold=0.25,
                            iou_threshold=0.5
                        )
                        
                        # Rulează experiment
                        try:
                            result = self._run_single_experiment(config, image_name)
                            self.results.append(result)
                        except Exception as e:
                            logger.error(f"❌ Eroare experiment {config.method}/{config.model_name}: {e}")
                        
                        pbar.update(1)
        
        pbar.close()
        
        logger.info(f"✅ Benchmark completat: {len(self.results)} experimente")
        
        # Salvează rezultate
        self._save_results()
        
        # Generează rapoarte
        self._generate_reports()
    
    def _run_single_experiment(
        self,
        config: ExperimentConfig,
        image_name: str
    ) -> Dict:
        """Rulează un singur experiment"""
        logger.info(f"🔬 Experiment: {config.method} + {config.model_name}")
        
        # Crează pipeline
        pipeline = Detection360Pipeline(config)
        
        # Procesează
        start_time = time.time()
        results = pipeline.process_image(config.input_image)
        total_time = time.time() - start_time
        
        # Extract metrics
        metrics = results['metrics']
        
        # Compilează rezultate
        experiment_result = {
            'image_name': image_name,
            'method': config.method,
            'model': config.model_name,
            'num_slices': config.num_slices if config.method == 'vertical_slice' else None,
            'face_size': config.face_size if config.method == 'cubemap' else None,
            'num_detections': metrics['num_detections'],
            'total_time': total_time,
            'fps': metrics['fps'],
            'preprocess_time': metrics['preprocess_time'],
            'detect_time': metrics['detect_time'],
            'merge_time': metrics['merge_time'],
            'num_tiles': metrics['num_tiles'],
            'avg_time_per_tile': metrics['avg_time_per_tile']
        }
        
        # Salvează vizualizare
        output_path = Path(config.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        import cv2
        cv2.imwrite(
            str(output_path / 'result.jpg'),
            results['visualization']
        )
        
        return experiment_result
    
    def _save_results(self):
        """Salvează rezultate ca JSON și CSV"""
        # JSON
        json_file = self.output_dir / 'results.json'
        with open(json_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # CSV
        df = pd.DataFrame(self.results)
        csv_file = self.output_dir / 'results.csv'
        df.to_csv(csv_file, index=False)
        
        logger.info(f"✅ Rezultate salvate: {json_file}, {csv_file}")
    
    def _generate_reports(self):
        """Generează rapoarte vizuale și statistice"""
        df = pd.DataFrame(self.results)
        
        if df.empty:
            logger.warning("Nu există rezultate pentru raport")
            return
        
        # 1. FPS Comparison
        self._plot_fps_comparison(df)
        
        # 2. Time Breakdown
        self._plot_time_breakdown(df)
        
        # 3. Detections Comparison
        self._plot_detections_comparison(df)
        
        # 4. Scalability Analysis
        self._plot_scalability(df)
        
        # 5. Summary Statistics
        self._generate_summary_stats(df)
        
        logger.info("✅ Rapoarte generate")
    
    def _plot_fps_comparison(self, df: pd.DataFrame):
        """Plot FPS per metodă și model"""
        plt.figure(figsize=(12, 6))
        
        # Group by method și model
        grouped = df.groupby(['method', 'model'])['fps'].mean().reset_index()
        
        sns.barplot(data=grouped, x='method', y='fps', hue='model')
        plt.title('FPS Comparison: Methods vs Models', fontsize=16, fontweight='bold')
        plt.ylabel('FPS (higher is better)', fontsize=12)
        plt.xlabel('Preprocessing Method', fontsize=12)
        plt.legend(title='YOLO Model')
        plt.tight_layout()
        
        plt.savefig(self.output_dir / 'fps_comparison.png', dpi=300)
        plt.close()
    
    def _plot_time_breakdown(self, df: pd.DataFrame):
        """Plot breakdown al timpilor"""
        plt.figure(figsize=(14, 6))
        
        # Calculează medie per metodă
        time_cols = ['preprocess_time', 'detect_time', 'merge_time']
        grouped = df.groupby('method')[time_cols].mean()
        
        grouped.plot(kind='bar', stacked=True)
        plt.title('Time Breakdown by Method', fontsize=16, fontweight='bold')
        plt.ylabel('Time (seconds)', fontsize=12)
        plt.xlabel('Preprocessing Method', fontsize=12)
        plt.legend(['Preprocess', 'Detection', 'Merge'], loc='upper right')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        plt.savefig(self.output_dir / 'time_breakdown.png', dpi=300)
        plt.close()
    
    def _plot_detections_comparison(self, df: pd.DataFrame):
        """Plot număr de detecții"""
        plt.figure(figsize=(12, 6))
        
        sns.boxplot(data=df, x='method', y='num_detections', hue='model')
        plt.title('Number of Detections: Methods vs Models', 
                 fontsize=16, fontweight='bold')
        plt.ylabel('Number of Detections', fontsize=12)
        plt.xlabel('Preprocessing Method', fontsize=12)
        plt.legend(title='YOLO Model')
        plt.tight_layout()
        
        plt.savefig(self.output_dir / 'detections_comparison.png', dpi=300)
        plt.close()
    
    def _plot_scalability(self, df: pd.DataFrame):
        """Plot scalabilitate cu num_tiles"""
        plt.figure(figsize=(12, 6))
        
        # Filter pentru vertical_slice care are num_slices variabil
        df_vs = df[df['method'] == 'vertical_slice'].copy()
        
        if not df_vs.empty:
            sns.scatterplot(data=df_vs, x='num_tiles', y='total_time', 
                           hue='model', style='model', s=100)
            plt.title('Scalability: Processing Time vs Number of Tiles', 
                     fontsize=16, fontweight='bold')
            plt.ylabel('Total Processing Time (s)', fontsize=12)
            plt.xlabel('Number of Tiles', fontsize=12)
            plt.legend(title='YOLO Model')
            plt.tight_layout()
            
            plt.savefig(self.output_dir / 'scalability.png', dpi=300)
        
        plt.close()
    
    def _generate_summary_stats(self, df: pd.DataFrame):
        """Generează statistici rezumative"""
        summary = []
        
        for method in df['method'].unique():
            for model in df['model'].unique():
                subset = df[(df['method'] == method) & (df['model'] == model)]
                
                if subset.empty:
                    continue
                
                stats = {
                    'Method': method,
                    'Model': model,
                    'Avg FPS': f"{subset['fps'].mean():.2f}",
                    'Avg Time (s)': f"{subset['total_time'].mean():.3f}",
                    'Avg Detections': f"{subset['num_detections'].mean():.1f}",
                    'Avg Tiles': f"{subset['num_tiles'].mean():.1f}",
                }
                summary.append(stats)
        
        summary_df = pd.DataFrame(summary)
        
        # Salvează ca Markdown
        md_file = self.output_dir / 'summary.md'
        with open(md_file, 'w') as f:
            f.write("# 360° Detection Benchmark - Summary\n\n")
            f.write(summary_df.to_markdown(index=False))
        
        # Salvează ca CSV
        summary_df.to_csv(self.output_dir / 'summary.csv', index=False)
        
        logger.info(f"✅ Summary salvat: {md_file}")


def main():
    """Main pentru rulare benchmark"""
    # Listă imagini de test
    test_images = [
        'data/samples/test_360_1.jpg',
        'data/samples/test_360_2.jpg',
    ]
    
    # Verifică că există imagini
    existing_images = [img for img in test_images if Path(img).exists()]
    
    if not existing_images:
        logger.error("❌ Nu există imagini de test în data/samples/")
        logger.info("💡 Adaugă imagini 360° în data/samples/ sau rulează cu imagini dummy")
        
        # Crează imagini dummy pentru testare
        import cv2
        import numpy as np
        
        samples_dir = Path('data/samples')
        samples_dir.mkdir(parents=True, exist_ok=True)
        
        for i in range(2):
            dummy_img = np.random.randint(0, 255, (1024, 2048, 3), dtype=np.uint8)
            # Adaugă unele forme pentru testare
            cv2.circle(dummy_img, (1024, 512), 100, (0, 255, 0), -1)
            cv2.rectangle(dummy_img, (500, 300), (700, 500), (255, 0, 0), 5)
            
            output_path = samples_dir / f'test_360_{i+1}.jpg'
            cv2.imwrite(str(output_path), dummy_img)
            logger.info(f"✅ Creat imagine dummy: {output_path}")
            existing_images.append(str(output_path))
    
    # Crează benchmark
    benchmark = Benchmark360(output_dir='results/benchmark')
    
    # Rulează experimente
    benchmark.run_experiments(
        test_images=existing_images,
        methods=['vertical_slice', 'cubemap'],
        models=['yolo11n.pt'],  # Folosește doar nano pentru testare rapidă
        num_slices_variants=[4, 6],
        face_sizes=[512]
    )
    
    print("\n" + "="*60)
    print("✅ BENCHMARK COMPLET!")
    print("="*60)
    print(f"Rezultate salvate în: results/benchmark/")
    print("  - results.json")
    print("  - results.csv")
    print("  - summary.md")
    print("  - fps_comparison.png")
    print("  - time_breakdown.png")
    print("  - detections_comparison.png")
    print("  - scalability.png")
    print("="*60)


if __name__ == '__main__':
    main()
