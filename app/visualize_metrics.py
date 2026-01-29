"""
Advanced Confusion Matrix and Metrics Visualizer
Creates comprehensive visualizations for model performance analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, auc
)
import json
import os
from pathlib import Path

# Styling
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


class MetricsVisualizer:
    """Class for creating comprehensive metric visualizations"""
    
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def plot_confusion_matrix(self, y_true, y_pred, normalize=False, title='Confusion Matrix'):
        """
        Plot confusion matrix with optional normalization
        """
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2%'
            title += ' (Normalized)'
        else:
            fmt = 'd'
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                   xticklabels=['Non-Violence', 'Violence'],
                   yticklabels=['Non-Violence', 'Violence'],
                   cbar_kws={'label': 'Count' if not normalize else 'Proportion'},
                   ax=ax)
        
        plt.title(title, fontsize=16, fontweight='bold', pad=20)
        plt.ylabel('True Label', fontsize=12, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
        
        # Add text annotations for cell meanings
        if not normalize:
            tn, fp, fn, tp = cm.ravel()
            ax.text(0.5, 0.25, 'TN', fontsize=8, color='gray', 
                   ha='center', transform=ax.transAxes)
            ax.text(0.5, 0.75, 'FN', fontsize=8, color='gray', 
                   ha='center', transform=ax.transAxes)
            ax.text(1.5, 0.25, 'FP', fontsize=8, color='gray', 
                   ha='center', transform=ax.transAxes)
            ax.text(1.5, 0.75, 'TP', fontsize=8, color='gray', 
                   ha='center', transform=ax.transAxes)
        
        plt.tight_layout()
        filename = 'confusion_matrix_normalized.png' if normalize else 'confusion_matrix.png'
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved: {filename}")
    
    def plot_roc_curve(self, y_true, y_scores, title='ROC Curve'):
        """Plot ROC curve with AUC"""
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        # Find optimal threshold (Youden's J statistic)
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        optimal_threshold = thresholds[optimal_idx]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot ROC curve
        ax.plot(fpr, tpr, color='darkorange', lw=2.5, 
               label=f'ROC Curve (AUC = {roc_auc:.3f})')
        
        # Plot random classifier
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
               label='Random Classifier')
        
        # Mark optimal point
        ax.scatter(fpr[optimal_idx], tpr[optimal_idx], color='red', s=100, 
                  zorder=5, label=f'Optimal Threshold = {optimal_threshold:.3f}')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Positive Rate (Recall)', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.legend(loc="lower right", fontsize=10)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'roc_curve.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved: roc_curve.png")
        
        return optimal_threshold
    
    def plot_precision_recall_curve(self, y_true, y_scores, title='Precision-Recall Curve'):
        """Plot Precision-Recall curve"""
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        pr_auc = auc(recall, precision)
        
        # Find F1-optimal threshold
        f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-10)
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        ax.plot(recall, precision, color='blue', lw=2.5, 
               label=f'PR Curve (AUC = {pr_auc:.3f})')
        
        # Mark optimal F1 point
        ax.scatter(recall[optimal_idx], precision[optimal_idx], color='red', s=100, 
                  zorder=5, label=f'Optimal F1 Threshold = {optimal_threshold:.3f}')
        
        # Baseline (prevalence)
        baseline = np.sum(y_true) / len(y_true)
        ax.axhline(y=baseline, color='gray', linestyle='--', lw=2, 
                  label=f'Baseline (Prevalence = {baseline:.3f})')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall', fontsize=12, fontweight='bold')
        ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.legend(loc="best", fontsize=10)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'precision_recall_curve.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved: precision_recall_curve.png")
        
        return optimal_threshold
    
    def plot_metric_comparison(self, metrics_dict, title='Performance Metrics Comparison'):
        """Bar chart comparing different metrics"""
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'F2-Score']
        metric_keys = ['accuracy', 'precision', 'recall', 'f1_score', 'f2_score']
        
        values = [metrics_dict.get(key, 0) for key in metric_keys]
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(metric_names, values, color=colors, alpha=0.8, edgecolor='black', lw=1.5)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.3f}',
                   ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        ax.set_ylim([0, 1.1])
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.grid(axis='y', alpha=0.3)
        
        # Add horizontal reference lines
        ax.axhline(y=0.7, color='orange', linestyle='--', alpha=0.5, label='Good (0.70)')
        ax.axhline(y=0.85, color='green', linestyle='--', alpha=0.5, label='Excellent (0.85)')
        ax.legend(loc='upper right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'metrics_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved: metrics_comparison.png")
    
    def plot_threshold_analysis(self, y_true, y_scores, current_threshold=0.70):
        """Comprehensive threshold analysis"""
        thresholds = np.linspace(0, 1, 200)
        
        precisions = []
        recalls = []
        f1_scores = []
        f2_scores = []
        accuracies = []
        
        for thresh in thresholds:
            y_pred = (y_scores >= thresh).astype(int)
            
            from sklearn.metrics import precision_score, recall_score, accuracy_score, fbeta_score
            precisions.append(precision_score(y_true, y_pred, zero_division=0))
            recalls.append(recall_score(y_true, y_pred, zero_division=0))
            
            prec = precisions[-1]
            rec = recalls[-1]
            f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
            f1_scores.append(f1)
            
            f2_scores.append(fbeta_score(y_true, y_pred, beta=2, zero_division=0))
            accuracies.append(accuracy_score(y_true, y_pred))
        
        # Find optimal thresholds
        optimal_f1_idx = np.argmax(f1_scores)
        optimal_f2_idx = np.argmax(f2_scores)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot 1: All metrics vs threshold
        ax1.plot(thresholds, precisions, label='Precision', linewidth=2.5, color='#e74c3c')
        ax1.plot(thresholds, recalls, label='Recall', linewidth=2.5, color='#2ecc71')
        ax1.plot(thresholds, f1_scores, label='F1-Score', linewidth=2.5, color='#3498db')
        ax1.plot(thresholds, f2_scores, label='F2-Score', linewidth=2.5, color='#9b59b6')
        ax1.plot(thresholds, accuracies, label='Accuracy', linewidth=2, color='#f39c12', linestyle='--')
        
        # Mark current threshold
        ax1.axvline(x=current_threshold, color='red', linestyle='--', linewidth=2,
                   label=f'Current ({current_threshold:.2f})', alpha=0.7)
        
        # Mark optimal thresholds
        ax1.axvline(x=thresholds[optimal_f1_idx], color='blue', linestyle=':', linewidth=2,
                   label=f'Optimal F1 ({thresholds[optimal_f1_idx]:.2f})', alpha=0.7)
        ax1.axvline(x=thresholds[optimal_f2_idx], color='purple', linestyle=':', linewidth=2,
                   label=f'Optimal F2 ({thresholds[optimal_f2_idx]:.2f})', alpha=0.7)
        
        ax1.set_xlabel('Threshold', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax1.set_title('Performance Metrics vs Threshold', fontsize=14, fontweight='bold')
        ax1.legend(loc='best', fontsize=9)
        ax1.grid(alpha=0.3)
        ax1.set_xlim([0, 1])
        ax1.set_ylim([0, 1.05])
        
        # Plot 2: Precision-Recall Trade-off
        ax2.plot(recalls, precisions, linewidth=3, color='#34495e')
        ax2.scatter(recalls[optimal_f1_idx], precisions[optimal_f1_idx], 
                   color='blue', s=150, zorder=5, label=f'Optimal F1', marker='o')
        ax2.scatter(recalls[optimal_f2_idx], precisions[optimal_f2_idx], 
                   color='purple', s=150, zorder=5, label=f'Optimal F2', marker='s')
        
        # Mark current threshold point
        current_idx = np.argmin(np.abs(thresholds - current_threshold))
        ax2.scatter(recalls[current_idx], precisions[current_idx], 
                   color='red', s=150, zorder=5, label=f'Current', marker='^')
        
        ax2.set_xlabel('Recall', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Precision', fontsize=12, fontweight='bold')
        ax2.set_title('Precision-Recall Trade-off', fontsize=14, fontweight='bold')
        ax2.legend(loc='best', fontsize=10)
        ax2.grid(alpha=0.3)
        ax2.set_xlim([0, 1.05])
        ax2.set_ylim([0, 1.05])
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'threshold_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved: threshold_analysis.png")
        
        return {
            'optimal_f1_threshold': thresholds[optimal_f1_idx],
            'optimal_f1_score': f1_scores[optimal_f1_idx],
            'optimal_f2_threshold': thresholds[optimal_f2_idx],
            'optimal_f2_score': f2_scores[optimal_f2_idx]
        }
    
    def plot_classification_report_heatmap(self, y_true, y_pred):
        """Visualize classification report as heatmap"""
        from sklearn.metrics import classification_report
        
        report = classification_report(y_true, y_pred, 
                                      target_names=['Non-Violence', 'Violence'],
                                      output_dict=True, zero_division=0)
        
        # Extract metrics for visualization
        classes = ['Non-Violence', 'Violence']
        metrics = ['precision', 'recall', 'f1-score']
        
        data = []
        for cls in classes:
            row = [report[cls][metric] for metric in metrics]
            data.append(row)
        
        data = np.array(data)
        
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(data, annot=True, fmt='.3f', cmap='RdYlGn', 
                   xticklabels=['Precision', 'Recall', 'F1-Score'],
                   yticklabels=classes,
                   vmin=0, vmax=1, cbar_kws={'label': 'Score'},
                   ax=ax)
        
        plt.title('Classification Report Heatmap', fontsize=16, fontweight='bold', pad=20)
        plt.ylabel('Class', fontsize=12, fontweight='bold')
        plt.xlabel('Metric', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'classification_report_heatmap.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved: classification_report_heatmap.png")
    
    def create_comprehensive_dashboard(self, y_true, y_scores, y_pred, 
                                      current_threshold=0.70):
        """Create a comprehensive 4-panel dashboard"""
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # Panel 1: Confusion Matrix
        ax1 = fig.add_subplot(gs[0, 0])
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Non-V', 'Violence'],
                   yticklabels=['Non-V', 'Violence'],
                   ax=ax1, cbar=False)
        ax1.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        ax1.set_ylabel('True Label', fontweight='bold')
        ax1.set_xlabel('Predicted Label', fontweight='bold')
        
        # Panel 2: ROC Curve
        ax2 = fig.add_subplot(gs[0, 1])
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        ax2.plot(fpr, tpr, color='darkorange', lw=2.5, label=f'AUC = {roc_auc:.3f}')
        ax2.plot([0, 1], [0, 1], 'navy', lw=2, linestyle='--')
        ax2.set_xlabel('False Positive Rate', fontweight='bold')
        ax2.set_ylabel('True Positive Rate', fontweight='bold')
        ax2.set_title('ROC Curve', fontsize=14, fontweight='bold')
        ax2.legend(loc='lower right')
        ax2.grid(alpha=0.3)
        
        # Panel 3: Precision-Recall Curve
        ax3 = fig.add_subplot(gs[1, 0])
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        pr_auc = auc(recall, precision)
        ax3.plot(recall, precision, color='blue', lw=2.5, label=f'AUC = {pr_auc:.3f}')
        ax3.set_xlabel('Recall', fontweight='bold')
        ax3.set_ylabel('Precision', fontweight='bold')
        ax3.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
        ax3.legend(loc='best')
        ax3.grid(alpha=0.3)
        
        # Panel 4: Metrics Comparison
        ax4 = fig.add_subplot(gs[1, 1])
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score
        
        metrics = {
            'Accuracy': accuracy_score(y_true, y_pred),
            'Precision': precision_score(y_true, y_pred, zero_division=0),
            'Recall': recall_score(y_true, y_pred, zero_division=0),
            'F1': f1_score(y_true, y_pred, zero_division=0),
            'F2': fbeta_score(y_true, y_pred, beta=2, zero_division=0)
        }
        
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
        bars = ax4.bar(metrics.keys(), metrics.values(), color=colors, alpha=0.8, edgecolor='black')
        
        for bar, value in zip(bars, metrics.values()):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        ax4.set_ylim([0, 1.1])
        ax4.set_ylabel('Score', fontweight='bold')
        ax4.set_title('Performance Metrics', fontsize=14, fontweight='bold')
        ax4.grid(axis='y', alpha=0.3)
        
        # Overall title
        fig.suptitle('Violence Detection Model - Performance Dashboard', 
                    fontsize=18, fontweight='bold', y=0.98)
        
        plt.savefig(os.path.join(self.output_dir, 'comprehensive_dashboard.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved: comprehensive_dashboard.png")


def demo_visualizations():
    """
    Demo function that creates sample visualizations
    Replace with your actual y_true, y_scores, y_pred from evaluation
    """
    # Example: Load from evaluation results
    results_file = r"C:\Users\akshi\OneDrive\Desktop\Akshit\Violence-Detection-System\evaluation_results\evaluation_results.json"
    
    if os.path.exists(results_file):
        print(f"📁 Loading results from: {results_file}")
        with open(results_file, 'r') as f:
            data = json.load(f)
        
        # Extract data (you'll need to parse based on your actual structure)
        print("⚠️  Please run evaluate.py first to generate evaluation data")
        print("   Then this script can visualize those results")
        return
    
    # Otherwise, create sample data for demonstration
    print("📊 Creating sample visualizations with synthetic data...")
    np.random.seed(42)
    n_samples = 100
    
    y_true = np.random.randint(0, 2, n_samples)
    y_scores = np.random.rand(n_samples)
    y_pred = (y_scores > 0.7).astype(int)
    
    output_dir = r"C:\Users\akshi\OneDrive\Desktop\Akshit\Violence-Detection-System\visualizations"
    
    visualizer = MetricsVisualizer(output_dir)
    
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60 + "\n")
    
    # Create all visualizations
    visualizer.plot_confusion_matrix(y_true, y_pred)
    visualizer.plot_confusion_matrix(y_true, y_pred, normalize=True)
    visualizer.plot_roc_curve(y_true, y_scores)
    visualizer.plot_precision_recall_curve(y_true, y_scores)
    
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
        'f2_score': fbeta_score(y_true, y_pred, beta=2, zero_division=0)
    }
    
    visualizer.plot_metric_comparison(metrics)
    visualizer.plot_threshold_analysis(y_true, y_scores)
    visualizer.plot_classification_report_heatmap(y_true, y_pred)
    visualizer.create_comprehensive_dashboard(y_true, y_scores, y_pred)
    
    print(f"\n✅ All visualizations saved to: {output_dir}")
    print("\n📌 To use with real data:")
    print("   1. Run evaluate.py to generate evaluation results")
    print("   2. Modify this script to load your actual y_true, y_scores, y_pred")


if __name__ == "__main__":
    demo_visualizations()
