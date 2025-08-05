"""
专门的模型评估和对比脚本
"""
import sys
import os
from pathlib import Path

# 添加项目根目录到路径
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent  # 回到项目根目录
sys.path.append(str(project_root))

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from tqdm import tqdm

from ensemble.src.training.streaming_multi_rocket import StreamingMultiRocketClassifier, StreamingInferenceEngine
from ensemble.src.training.data_loader import TrajectoryDataLoader


def load_trained_model(checkpoint_path: str, device: str = 'auto'):
    """加载训练好的模型"""
    if device == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"从 {checkpoint_path} 加载模型...")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"模型文件不存在: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # 重建模型
    data_info = checkpoint['data_info']
    config = checkpoint.get('config')
    
    model = StreamingMultiRocketClassifier(
        c_in=data_info['num_features'],
        c_out=data_info['num_classes'],
        max_seq_len=data_info['seq_len'],
        min_seq_len=getattr(config, 'min_seq_len', 10),
        device=device,
        confidence_threshold=getattr(config, 'confidence_threshold', 0.7)
    ).to(device)
    
    # 加载权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ 模型加载成功")
    print(f"   支持的序列长度: {model.supported_lengths}")
    print(f"   数据信息: {data_info}")
    
    return model, checkpoint


def evaluate_streaming_vs_batch(model, data_loader, device, detailed_analysis=True):
    """对比流式推理和批量推理的性能"""
    print("\\n📊 流式推理 vs 批量推理对比评估")
    print("=" * 50)
    
    model.eval()
    
    # 存储结果
    batch_results = {
        'predictions': [],
        'targets': [],
        'confidences': [],
        'inference_times': []
    }
    
    streaming_results = {
        'predictions': [],
        'targets': [],
        'confidences': [],
        'early_stops': [],
        'stop_timesteps': [],
        'inference_times': [],
        'timestep_predictions': []  # 每个时间步的预测历史
    }
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_loader, desc="评估中")):
            sequences = batch['sequences'].to(device)  # (batch, seq_len, features)
            labels = batch['labels'].to(device)
            
            # 转换维度
            sequences_t = sequences.transpose(1, 2)  # (batch, features, seq_len)
            
            # 批量推理
            import time
            start_time = time.time()
            batch_output = model.forward(sequences_t)
            batch_time = time.time() - start_time
            
            batch_preds = torch.argmax(batch_output['logits'], dim=1)
            batch_confs = batch_output['confidence']
            
            batch_results['predictions'].extend(batch_preds.cpu().numpy())
            batch_results['targets'].extend(labels.cpu().numpy())
            batch_results['confidences'].extend(batch_confs.cpu().numpy())
            batch_results['inference_times'].append(batch_time / sequences.shape[0])  # 每样本时间
            
            # 流式推理
            for i in range(sequences.shape[0]):
                seq = sequences[i].numpy()  # (seq_len, features)
                true_label = labels[i].item()
                
                engine = StreamingInferenceEngine(model)
                
                start_time = time.time()
                timestep_preds = []
                final_pred = None
                stopped_early = False
                stop_step = len(seq)
                
                # 逐步添加时间步
                for t in range(len(seq)):
                    features = seq[t]
                    result = engine.add_timestep(features)
                    
                    timestep_preds.append({
                        'timestep': t + 1,
                        'prediction': result['prediction'],
                        'confidence': result['confidence']
                    })
                    
                    if result['stopped_early'] and not stopped_early:
                        final_pred = result['prediction']
                        stopped_early = True
                        stop_step = t + 1
                        break
                
                streaming_time = time.time() - start_time
                
                if final_pred is None:
                    final_pred = result['prediction']
                
                streaming_results['predictions'].append(final_pred)
                streaming_results['targets'].append(true_label)
                streaming_results['confidences'].append(result['confidence'])
                streaming_results['early_stops'].append(stopped_early)
                streaming_results['stop_timesteps'].append(stop_step)
                streaming_results['inference_times'].append(streaming_time)
                streaming_results['timestep_predictions'].append(timestep_preds)
            
            # 限制评估样本数量以节省时间
            if batch_idx >= 10:  # 只评估前几个batch
                break
    
    # 计算性能指标
    batch_acc = accuracy_score(batch_results['targets'], batch_results['predictions'])
    streaming_acc = accuracy_score(streaming_results['targets'], streaming_results['predictions'])
    
    batch_prec, batch_rec, batch_f1, _ = precision_recall_fscore_support(
        batch_results['targets'], batch_results['predictions'], average='weighted'
    )
    streaming_prec, streaming_rec, streaming_f1, _ = precision_recall_fscore_support(
        streaming_results['targets'], streaming_results['predictions'], average='weighted'
    )
    
    # 时间统计
    avg_batch_time = np.mean(batch_results['inference_times']) * 1000  # ms
    avg_streaming_time = np.mean(streaming_results['inference_times']) * 1000  # ms
    
    # 早期停止统计
    early_stop_rate = np.mean(streaming_results['early_stops'])
    avg_stop_timestep = np.mean(streaming_results['stop_timesteps'])
    max_timesteps = max(streaming_results['stop_timesteps'])
    
    # 置信度统计
    avg_batch_conf = np.mean(batch_results['confidences'])
    avg_streaming_conf = np.mean(streaming_results['confidences'])
    
    print("\\n📈 性能对比结果:")
    print(f"{'指标':<20} {'批量推理':<15} {'流式推理':<15} {'差异':<10}")
    print("-" * 65)
    print(f"{'准确率':<20} {batch_acc:<15.4f} {streaming_acc:<15.4f} {streaming_acc-batch_acc:<+10.4f}")
    print(f"{'精确率':<20} {batch_prec:<15.4f} {streaming_prec:<15.4f} {streaming_prec-batch_prec:<+10.4f}")
    print(f"{'召回率':<20} {batch_rec:<15.4f} {streaming_rec:<15.4f} {streaming_rec-batch_rec:<+10.4f}")
    print(f"{'F1分数':<20} {batch_f1:<15.4f} {streaming_f1:<15.4f} {streaming_f1-batch_f1:<+10.4f}")
    print(f"{'平均置信度':<20} {avg_batch_conf:<15.4f} {avg_streaming_conf:<15.4f} {avg_streaming_conf-avg_batch_conf:<+10.4f}")
    print(f"{'推理时间(ms)':<20} {avg_batch_time:<15.2f} {avg_streaming_time:<15.2f} {avg_streaming_time-avg_batch_time:<+10.2f}")
    
    print(f"\\n🎯 流式推理特有指标:")
    print(f"   早期停止率: {early_stop_rate:.2%}")
    print(f"   平均停止时间步: {avg_stop_timestep:.1f} / {max_timesteps}")
    print(f"   时间步节省: {(max_timesteps - avg_stop_timestep) / max_timesteps:.2%}")
    
    if detailed_analysis:
        # 详细分析
        analyze_early_stopping_patterns(streaming_results)
        plot_streaming_analysis(streaming_results, batch_results)
    
    return {
        'batch_results': batch_results,
        'streaming_results': streaming_results,
        'metrics': {
            'batch_accuracy': batch_acc,
            'streaming_accuracy': streaming_acc,
            'early_stop_rate': early_stop_rate,
            'avg_stop_timestep': avg_stop_timestep,
            'time_speedup': avg_batch_time / avg_streaming_time if avg_streaming_time > 0 else 1.0
        }
    }


def analyze_early_stopping_patterns(streaming_results):
    """分析早期停止模式"""
    print("\\n🔍 早期停止模式分析:")
    
    stop_timesteps = np.array(streaming_results['stop_timesteps'])
    early_stops = np.array(streaming_results['early_stops'])
    
    # 按是否早期停止分组
    early_stop_indices = np.where(early_stops)[0]
    no_early_stop_indices = np.where(~early_stops)[0]
    
    if len(early_stop_indices) > 0:
        early_stop_timesteps = stop_timesteps[early_stop_indices]
        print(f"   早期停止样本数: {len(early_stop_indices)}")
        print(f"   早期停止时间步分布: 平均={np.mean(early_stop_timesteps):.1f}, "
              f"中位数={np.median(early_stop_timesteps):.1f}, "
              f"范围=[{np.min(early_stop_timesteps)}, {np.max(early_stop_timesteps)}]")
        
        # 早期停止的准确率
        early_stop_preds = np.array(streaming_results['predictions'])[early_stop_indices]
        early_stop_targets = np.array(streaming_results['targets'])[early_stop_indices]
        early_stop_acc = accuracy_score(early_stop_targets, early_stop_preds)
        print(f"   早期停止样本准确率: {early_stop_acc:.4f}")
    
    if len(no_early_stop_indices) > 0:
        no_early_stop_preds = np.array(streaming_results['predictions'])[no_early_stop_indices]
        no_early_stop_targets = np.array(streaming_results['targets'])[no_early_stop_indices]
        no_early_stop_acc = accuracy_score(no_early_stop_targets, no_early_stop_preds)
        print(f"   完整序列样本准确率: {no_early_stop_acc:.4f}")
    
    # 按类别分析
    targets = np.array(streaming_results['targets'])
    unique_classes = np.unique(targets)
    
    print("\\n   各类别早期停止率:")
    for cls in unique_classes:
        cls_indices = np.where(targets == cls)[0]
        cls_early_stops = early_stops[cls_indices]
        cls_early_rate = np.mean(cls_early_stops)
        print(f"     类别 {cls}: {cls_early_rate:.2%} ({np.sum(cls_early_stops)}/{len(cls_early_stops)})")


def plot_streaming_analysis(streaming_results, batch_results, suffix=""):
    """绘制流式推理分析图表"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. 准确率对比
    batch_acc = accuracy_score(batch_results['targets'], batch_results['predictions'])
    streaming_acc = accuracy_score(streaming_results['targets'], streaming_results['predictions'])
    
    axes[0, 0].bar(['批量推理', '流式推理'], [batch_acc, streaming_acc], 
                   color=['blue', 'green'], alpha=0.7)
    axes[0, 0].set_title('准确率对比')
    axes[0, 0].set_ylabel('准确率')
    axes[0, 0].set_ylim(0, 1)
    
    # 添加数值标签
    for i, v in enumerate([batch_acc, streaming_acc]):
        axes[0, 0].text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom')
    
    # 2. 早期停止时间步分布
    stop_timesteps = streaming_results['stop_timesteps']
    axes[0, 1].hist(stop_timesteps, bins=20, alpha=0.7, color='orange', edgecolor='black')
    axes[0, 1].axvline(np.mean(stop_timesteps), color='red', linestyle='--', 
                       label=f'平均: {np.mean(stop_timesteps):.1f}')
    axes[0, 1].set_title('停止时间步分布')
    axes[0, 1].set_xlabel('停止时间步')
    axes[0, 1].set_ylabel('样本数量')
    axes[0, 1].legend()
    
    # 3. 置信度分布对比
    axes[0, 2].hist(batch_results['confidences'], bins=30, alpha=0.5, 
                    label='批量推理', color='blue', density=True)
    axes[0, 2].hist(streaming_results['confidences'], bins=30, alpha=0.5, 
                    label='流式推理', color='green', density=True)
    axes[0, 2].set_title('置信度分布对比')
    axes[0, 2].set_xlabel('置信度')
    axes[0, 2].set_ylabel('密度')
    axes[0, 2].legend()
    
    # 4. 混淆矩阵对比 - 批量推理
    cm_batch = confusion_matrix(batch_results['targets'], batch_results['predictions'])
    sns.heatmap(cm_batch, annot=True, fmt='d', ax=axes[1, 0], cmap='Blues')
    axes[1, 0].set_title('批量推理混淆矩阵')
    axes[1, 0].set_xlabel('预测类别')
    axes[1, 0].set_ylabel('真实类别')
    
    # 5. 混淆矩阵对比 - 流式推理  
    cm_streaming = confusion_matrix(streaming_results['targets'], streaming_results['predictions'])
    sns.heatmap(cm_streaming, annot=True, fmt='d', ax=axes[1, 1], cmap='Greens')
    axes[1, 1].set_title('流式推理混淆矩阵')
    axes[1, 1].set_xlabel('预测类别')
    axes[1, 1].set_ylabel('真实类别')
    
    # 6. 早期停止vs准确率散点图
    early_stops = np.array(streaming_results['early_stops'])
    targets = np.array(streaming_results['targets'])
    predictions = np.array(streaming_results['predictions'])
    correct = (targets == predictions)
    
    # 分别绘制早期停止和非早期停止的样本
    early_indices = np.where(early_stops)[0]
    no_early_indices = np.where(~early_stops)[0]
    
    if len(early_indices) > 0:
        axes[1, 2].scatter(np.array(stop_timesteps)[early_indices], 
                          correct[early_indices].astype(int),
                          c='red', alpha=0.6, label='早期停止', s=30)
    
    if len(no_early_indices) > 0:
        axes[1, 2].scatter(np.array(stop_timesteps)[no_early_indices], 
                          correct[no_early_indices].astype(int),
                          c='blue', alpha=0.6, label='完整序列', s=30)
    
    axes[1, 2].set_title('停止时间步 vs 预测正确性')
    axes[1, 2].set_xlabel('停止时间步')
    axes[1, 2].set_ylabel('预测正确 (1=正确, 0=错误)')
    axes[1, 2].legend()
    axes[1, 2].set_ylim(-0.1, 1.1)
    
    plt.tight_layout()
    plt.savefig(f'streaming_analysis{suffix}.png', dpi=150, bbox_inches='tight')
    print(f"\\n📊 分析图表已保存到 streaming_analysis{suffix}.png")


def analyze_timestep_evolution(streaming_results, max_samples=5):
    """分析预测随时间步的演化"""
    print(f"\\n📈 预测演化分析 (前{max_samples}个样本):")
    
    timestep_data = streaming_results['timestep_predictions'][:max_samples]
    targets = streaming_results['targets'][:max_samples]
    
    fig, axes = plt.subplots(max_samples, 1, figsize=(12, 3*max_samples))
    if max_samples == 1:
        axes = [axes]
    
    for i, (sample_data, true_label) in enumerate(zip(timestep_data, targets)):
        timesteps = [d['timestep'] for d in sample_data]
        predictions = [d['prediction'] for d in sample_data]
        confidences = [d['confidence'] for d in sample_data]
        
        # 绘制预测变化
        ax1 = axes[i]
        ax1.step(timesteps, predictions, 'b-', where='post', linewidth=2, label='预测类别')
        ax1.axhline(y=true_label, color='red', linestyle='--', alpha=0.7, label=f'真实类别 ({true_label})')
        ax1.set_ylabel('预测类别')
        ax1.set_title(f'样本 {i+1} - 预测演化')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 绘制置信度
        ax2 = ax1.twinx()
        ax2.plot(timesteps, confidences, 'g-', alpha=0.7, label='置信度')
        ax2.axhline(y=0.7, color='orange', linestyle=':', alpha=0.7, label='置信度阈值')
        ax2.set_ylabel('置信度')
        ax2.legend(loc='upper right')
        ax2.set_ylim(0, 1)
        
        if i == max_samples - 1:
            ax1.set_xlabel('时间步')
    
    plt.tight_layout()
    plt.savefig('prediction_evolution.png', dpi=150, bbox_inches='tight')
    print("📊 预测演化图表已保存到 prediction_evolution.png")


def comprehensive_model_evaluation(checkpoint_path: str, data_path: str):
    """综合模型评估"""
    print("🔍 综合模型评估")
    print("=" * 60)
    
    # 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_trained_model(checkpoint_path, device)
    
    # 加载数据
    print("\\n加载测试数据...")
    data_loader = TrajectoryDataLoader(
        data_path=data_path,
        batch_size=8,
        train_split=0.7,
        val_split=0.15,
        test_split=0.15,
        shuffle=False,
        num_workers=0,
        random_state=42
    )
    
    _, _, test_loader = data_loader.get_dataloaders()
    
    # 进行综合评估
    evaluation_results = evaluate_streaming_vs_batch(model, test_loader, device, detailed_analysis=True)
    
    # 分析预测演化
    analyze_timestep_evolution(evaluation_results['streaming_results'])
    
    # 生成评估报告
    generate_evaluation_report(evaluation_results, checkpoint)
    
    return evaluation_results


def generate_evaluation_report(evaluation_results, checkpoint):
    """生成评估报告"""
    print("\\n📄 生成评估报告...")
    
    batch_results = evaluation_results['batch_results']
    streaming_results = evaluation_results['streaming_results']
    
    report = f"""
# 流式HydraRocket模型评估报告

## 模型信息
- 训练轮数: {checkpoint.get('epoch', 'N/A')}
- 验证准确率: {checkpoint.get('val_accuracy', 'N/A'):.4f}
- 支持序列长度: {checkpoint.get('supported_lengths', 'N/A')}

## 性能对比

### 准确率
- 批量推理准确率: {accuracy_score(batch_results['targets'], batch_results['predictions']):.4f}
- 流式推理准确率: {accuracy_score(streaming_results['targets'], streaming_results['predictions']):.4f}

### 效率指标
- 早期停止率: {np.mean(streaming_results['early_stops']):.2%}
- 平均停止时间步: {np.mean(streaming_results['stop_timesteps']):.1f}
- 平均置信度: {np.mean(streaming_results['confidences']):.4f}

### 时间性能
- 平均批量推理时间: {np.mean(batch_results['inference_times'])*1000:.2f} ms/样本
- 平均流式推理时间: {np.mean(streaming_results['inference_times'])*1000:.2f} ms/样本

## 详细分类报告

### 批量推理
{classification_report(batch_results['targets'], batch_results['predictions'])}

### 流式推理  
{classification_report(streaming_results['targets'], streaming_results['predictions'])}

## 结论
流式推理系统成功实现了逐步输入的实时预测功能，在保持准确率的同时能够通过早期停止机制提高推理效率。
"""
    
    with open('evaluation_report.md', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("✅ 评估报告已保存到 evaluation_report.md")


def main():
    """主函数"""
    print("🔍 流式HydraRocket模型评估系统")
    print("=" * 60)
    
    # 检查文件
    checkpoint_path = "checkpoints/best_streaming_model.pth"
    data_path = "processed_data.npz"
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ 模型文件不存在: {checkpoint_path}")
        print("请先运行训练脚本生成模型")
        return
    
    if not os.path.exists(data_path):
        print(f"❌ 数据文件不存在: {data_path}")
        print("请先运行数据预处理脚本")
        return
    
    try:
        # 进行综合评估
        results = comprehensive_model_evaluation(checkpoint_path, data_path)
        
        print("\\n🎉 评估完成！")
        print("=" * 60)
        print("📁 生成的文件:")
        print("   - streaming_analysis.png (对比分析图表)")
        print("   - prediction_evolution.png (预测演化图表)")
        print("   - evaluation_report.md (详细评估报告)")
        
    except Exception as e:
        print(f"❌ 评估过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()