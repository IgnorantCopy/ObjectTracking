"""
评估和可视化工具
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import torch
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


class TrainingVisualizer:
    """训练过程可视化"""
    
    def __init__(self, save_dir: str = "plots"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置中文字体和样式
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Liberation Sans']
        plt.rcParams['axes.unicode_minus'] = False
        sns.set_style("whitegrid")
        sns.set_palette("husl")
    
    def plot_training_history(self, 
                            train_history: Dict[str, List[float]], 
                            val_history: Dict[str, List[float]],
                            save_name: str = "training_history.png"):
        """绘制训练历史"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # 绘制损失曲线
        epochs = range(1, len(train_history['loss']) + 1)
        ax1.plot(epochs, train_history['loss'], 'b-', label='Training Loss', linewidth=2)
        ax1.plot(epochs, val_history['loss'], 'r-', label='Validation Loss', linewidth=2)
        ax1.set_title('Model Loss', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 绘制准确率曲线
        ax2.plot(epochs, train_history['accuracy'], 'b-', label='Training Accuracy', linewidth=2)
        ax2.plot(epochs, val_history['accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
        ax2.set_title('Model Accuracy', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"训练历史图已保存: {self.save_dir / save_name}")
    
    def plot_confusion_matrix(self, 
                            confusion_matrix: np.ndarray,
                            class_names: Optional[List[str]] = None,
                            normalize: bool = True,
                            save_name: str = "confusion_matrix.png"):
        """绘制混淆矩阵"""
        if normalize:
            cm = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
            title = 'Normalized Confusion Matrix'
            fmt = '.2f'
        else:
            cm = confusion_matrix
            title = 'Confusion Matrix'
            fmt = 'd'
        
        if class_names is None:
            class_names = [f'Class {i}' for i in range(len(confusion_matrix))]
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names,
                   cbar_kws={'label': 'Value'})
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"混淆矩阵已保存: {self.save_dir / save_name}")
    
    def plot_class_distribution(self, 
                               labels: np.ndarray,
                               class_names: Optional[List[str]] = None,
                               save_name: str = "class_distribution.png"):
        """绘制类别分布"""
        unique_labels, counts = np.unique(labels, return_counts=True)
        
        if class_names is None:
            class_names = [f'Class {i}' for i in unique_labels]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(class_names, counts, alpha=0.8, color=sns.color_palette("husl", len(unique_labels)))
        
        # 添加数值标签
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*max(counts),
                    str(count), ha='center', va='bottom', fontweight='bold')
        
        plt.title('Class Distribution', fontsize=14, fontweight='bold')
        plt.xlabel('Class')
        plt.ylabel('Number of Samples')
        plt.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"类别分布图已保存: {self.save_dir / save_name}")
    
    def plot_feature_importance(self, 
                              feature_importance: np.ndarray,
                              feature_names: Optional[List[str]] = None,
                              top_k: int = 20,
                              save_name: str = "feature_importance.png"):
        """绘制特征重要性（如果模型支持）"""
        if feature_names is None:
            feature_names = [f'Feature {i}' for i in range(len(feature_importance))]
        
        # 获取top_k最重要的特征
        indices = np.argsort(feature_importance)[-top_k:]
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(indices)), feature_importance[indices], alpha=0.8)
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('Importance')
        plt.title(f'Top {top_k} Feature Importance', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"特征重要性图已保存: {self.save_dir / save_name}")
    
    def plot_tsne_visualization(self, 
                               features: np.ndarray,
                               labels: np.ndarray,
                               class_names: Optional[List[str]] = None,
                               save_name: str = "tsne_visualization.png"):
        """使用t-SNE进行特征可视化"""
        if features.shape[0] > 5000:
            # 如果样本太多，随机采样
            indices = np.random.choice(features.shape[0], 5000, replace=False)
            features = features[indices]
            labels = labels[indices]
        
        print("正在计算t-SNE...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        features_2d = tsne.fit_transform(features)
        
        if class_names is None:
            class_names = [f'Class {i}' for i in np.unique(labels)]
        
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], 
                           c=labels, cmap='tab10', alpha=0.7, s=20)
        
        # 添加图例
        handles = []
        for i, class_name in enumerate(class_names):
            handles.append(plt.Line2D([0], [0], marker='o', color='w', 
                                    markerfacecolor=plt.cm.tab10(i), markersize=8, 
                                    label=class_name))
        plt.legend(handles=handles, loc='best')
        
        plt.title('t-SNE Visualization of Features', fontsize=14, fontweight='bold')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"t-SNE可视化已保存: {self.save_dir / save_name}")


class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self, class_names: Optional[List[str]] = None):
        self.class_names = class_names or ['类别1', '类别2', '类别3', '类别4']
    
    def evaluate_predictions(self, 
                           y_true: np.ndarray, 
                           y_pred: np.ndarray,
                           y_prob: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """全面评估预测结果"""
        from sklearn.metrics import (
            accuracy_score, precision_recall_fscore_support,
            classification_report, confusion_matrix,
            roc_auc_score, roc_curve
        )
        
        results = {}
        
        # 基本指标
        results['accuracy'] = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        results['precision'] = precision
        results['recall'] = recall
        results['f1_score'] = f1
        
        # 每个类别的指标
        precision_per_class, recall_per_class, f1_per_class, support = \
            precision_recall_fscore_support(y_true, y_pred, average=None)
        
        results['per_class_metrics'] = {
            'precision': precision_per_class,
            'recall': recall_per_class,
            'f1_score': f1_per_class,
            'support': support
        }
        
        # 混淆矩阵
        results['confusion_matrix'] = confusion_matrix(y_true, y_pred)
        
        # 分类报告
        results['classification_report'] = classification_report(
            y_true, y_pred, target_names=self.class_names
        )
        
        # ROC AUC (如果有概率预测)
        if y_prob is not None:
            try:
                if len(np.unique(y_true)) == 2:
                    # 二分类
                    results['roc_auc'] = roc_auc_score(y_true, y_prob[:, 1])
                else:
                    # 多分类
                    results['roc_auc'] = roc_auc_score(y_true, y_prob, multi_class='ovr')
            except Exception as e:
                print(f"计算ROC AUC时出错: {e}")
        
        return results
    
    def print_evaluation_report(self, results: Dict[str, Any]):
        """打印评估报告"""
        print("="*60)
        print("模型评估报告")
        print("="*60)
        
        print(f"总体准确率: {results['accuracy']:.4f}")
        print(f"加权精确率: {results['precision']:.4f}")
        print(f"加权召回率: {results['recall']:.4f}")
        print(f"加权F1分数: {results['f1_score']:.4f}")
        
        if 'roc_auc' in results:
            print(f"ROC AUC: {results['roc_auc']:.4f}")
        
        print("\n各类别详细指标:")
        print("-"*60)
        per_class = results['per_class_metrics']
        for i, class_name in enumerate(self.class_names):
            print(f"{class_name}:")
            print(f"  精确率: {per_class['precision'][i]:.4f}")
            print(f"  召回率: {per_class['recall'][i]:.4f}")
            print(f"  F1分数: {per_class['f1_score'][i]:.4f}")
            print(f"  样本数: {per_class['support'][i]}")
        
        print("\n分类报告:")
        print("-"*60)
        print(results['classification_report'])
    
    def create_comprehensive_report(self, 
                                  results: Dict[str, Any],
                                  save_path: str = "evaluation_report.txt"):
        """创建综合评估报告文件"""
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("模型评估报告\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"总体准确率: {results['accuracy']:.4f}\n")
            f.write(f"加权精确率: {results['precision']:.4f}\n")
            f.write(f"加权召回率: {results['recall']:.4f}\n")
            f.write(f"加权F1分数: {results['f1_score']:.4f}\n")
            
            if 'roc_auc' in results:
                f.write(f"ROC AUC: {results['roc_auc']:.4f}\n")
            
            f.write("\n各类别详细指标:\n")
            f.write("-"*60 + "\n")
            per_class = results['per_class_metrics']
            for i, class_name in enumerate(self.class_names):
                f.write(f"{class_name}:\n")
                f.write(f"  精确率: {per_class['precision'][i]:.4f}\n")
                f.write(f"  召回率: {per_class['recall'][i]:.4f}\n")
                f.write(f"  F1分数: {per_class['f1_score'][i]:.4f}\n")
                f.write(f"  样本数: {per_class['support'][i]}\n")
            
            f.write("\n分类报告:\n")
            f.write("-"*60 + "\n")
            f.write(results['classification_report'])
            
            f.write("\n混淆矩阵:\n")
            f.write("-"*60 + "\n")
            f.write(str(results['confusion_matrix']))
        
        print(f"评估报告已保存: {save_path}")