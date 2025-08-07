"""
专门的模型评估和对比脚本
"""
import time
import sys
import os
import re
import glob
from pathlib import Path
from typing import Tuple, Dict, Any

# 添加项目根目录到路径
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent  # 回到项目根目录
sys.path.append(str(project_root))

import torch
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from tqdm import tqdm

from src.training.streaming_multi_rocket import StreamingMultiRocketClassifier, StreamingInferenceEngine
from src.training.data_loader import TrajectoryDataLoader
from src.utils.config import TRACK_COLUMNS, DATA_ROOT, SEQ_LEN


def load_trained_model(checkpoint_path: str, device: str = 'auto'):
    """加载训练好的模型"""
    if device == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device == 'cpu':
        device = torch.device("cpu")
    elif device == 'cuda':
        device = torch.device("cuda")
    else:
        raise ValueError(f"不支持的设备: {device}")
    
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
        num_features=getattr(config, 'num_features', 10_000),
        dropout=getattr(config, 'dropout', 0.2),
        # confidence_threshold=getattr(config, 'confidence_threshold', 0.9)
        confidence_threshold=0.95
    ).to(device)
    
    # 加载权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ 模型加载成功")
    print(f"   支持的序列长度: {model.supported_lengths}")
    print(f"   数据信息: {data_info}")
    
    return model, checkpoint, device


def evaluate_streaming(model, data_loader, device, detailed_analysis=True):
    """对比流式推理和批量推理的性能"""
    print("=" * 50)
    
    model.eval()
    
    # 存储结果
    streaming_results = {
        'predictions': [],
        'labels': [],
        'batch_ids': [],
        'begin_time': [],
        'rates': [],
        'stop_timesteps': [],
        'inference_times': [],
        'timestep_predictions': []  # 每个时间步的预测历史
    }
    engine = StreamingInferenceEngine(model)

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_loader, desc="评估中")):
            sequences = batch['sequences'].to(device)  # (batch, seq_len, features)
            sequences = sequences.transpose(1, 2)
            labels = batch['labels'].to(device)
            batch_ids = batch['batch_ids']
            streaming_results['batch_ids'].extend(batch_ids)

            # 流式推理
            for i in range(sequences.shape[0]):
                engine.reset()
                seq = sequences[i]  # (features, seq_len)
                true_label = labels[i].item()

                start_time = time.time()
                timestep_preds = []
                # 逐步添加时间步
                is_begin = False
                for t in range(1, seq.shape[1] + 1):
                    features = seq[:, :t]
                    result = engine.add_timestep(features)

                    prediction = result['prediction']
                    seq[-1, t - 1] = prediction
                    timestep_preds.append(prediction)

                    if true_label == prediction and not is_begin:
                        is_begin = True
                        streaming_results['begin_time'].append(t + 1)

                if not is_begin:
                    streaming_results['begin_time'].append(len(seq))
                streaming_time = time.time() - start_time

                result = engine.get_final_prediction()
                final_pred = result['prediction']
                stop_step = result['stop_timestep']
                rate = result['rate']

                streaming_results['predictions'].append(final_pred)
                streaming_results['labels'].append(true_label)
                streaming_results['rates'].append(rate)
                streaming_results['stop_timesteps'].append(stop_step)
                streaming_results['inference_times'].append(streaming_time)
                streaming_results['timestep_predictions'].append(timestep_preds)

    # 计算性能指标
    streaming_accuracies = []
    predictions = np.array(streaming_results['predictions'])
    labels = np.array(streaming_results['labels'])
    rates = np.array(streaming_results['rates'])
    timestep_predictions = np.array(streaming_results['timestep_predictions'])
    for t in range(sequences.shape[2]):
        pred_t = timestep_predictions[:, t]
        streaming_accuracies.append(accuracy_score(labels, pred_t))
    avg_acc = 0
    for i in range(len(labels)):
        if predictions[i] == labels[i] and rates[i] >= 0.9:
            avg_acc += 1
    avg_acc /= len(labels)

    streaming_prec, streaming_rec, streaming_f1, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted'
    )
    
    # 时间统计
    avg_streaming_time = np.mean(streaming_results['inference_times']) * 1000  # ms
    
    # 早期停止统计
    avg_stop_timestep = np.mean(streaming_results['stop_timesteps'])
    avg_begin_timestep = np.mean(streaming_results['begin_time'])
    max_timesteps = max(streaming_results['stop_timesteps'])
    
    print(f"{'指标':<20} {'流式推理':<15}")
    print("-" * 65)
    print(f"{'准确率':<20} {avg_acc:<15.4f}")
    print(f"{'精确率':<20} {streaming_prec:<15.4f}")
    print(f"{'召回率':<20} {streaming_rec:<15.4f}")
    print(f"{'F1分数':<20} {streaming_f1:<15.4f}")
    print(f"{'推理时间(ms)':<20} {avg_streaming_time:<15.2f}")
    print(f"{'平均开始时间步':<20} {avg_begin_timestep:.1f}")
    print(f"{'平均停止时间步':<20} {avg_stop_timestep:.1f} / {max_timesteps}")

    if detailed_analysis:
        # 详细分析
        fig = go.Figure(data=[
            go.Scatter(x=np.arange(1, len(streaming_accuracies)+1), y=streaming_accuracies, mode='lines')
        ])
        fig.update_layout(
            title="准确率随时间步变化",
            xaxis_title="时间步",
            yaxis_title="准确率",
            legend_title="准确率"
        )
        fig.show()

    return {
        'streaming_results': streaming_results,
        'metrics': {
            'streaming_accuracy': streaming_prec,
            'avg_stop_timestep': avg_stop_timestep,
        }
    }


def export_results(results: Dict[str, Any], output_path: str, data_root: str):
    """导出结果"""
    timestep_predictions = results['streaming_results']['timestep_predictions']
    batch_ids = results['streaming_results']['batch_ids']
    track_files = glob.glob(os.path.join(data_root, "航迹/Tracks_*.txt"))
    for track_file in track_files:
        match_result = re.match(r"Tracks_(\d+)_(\d+)\.txt", os.path.basename(track_file))
        batch_id = match_result.group(1)
        num_points = int(match_result.group(2))
        if batch_id in batch_ids:
            df = pd.read_csv(track_file, encoding='gbk', header=0, names=TRACK_COLUMNS)
            timestep_prediction = timestep_predictions[batch_ids.index(batch_id)]
            if num_points <= SEQ_LEN:
                timestep_prediction = timestep_prediction[:num_points]
            else:
                timestep_prediction.extend([timestep_prediction[-1]] * (num_points - SEQ_LEN))
            df['识别结果'] = np.array(timestep_prediction) + 1
            df.to_csv(os.path.join(output_path, os.path.basename(track_file)), index=False, encoding='gbk')
    print(f"✅ 结果导出成功: {output_path}")


def comprehensive_model_evaluation(checkpoint_path: str):
    """综合模型评估"""
    print("🔍 综合模型评估")
    print("=" * 60)
    
    # 加载模型
    device = 'auto'
    model, checkpoint, device = load_trained_model(checkpoint_path, device)
    
    # 加载数据
    print("\n加载测试数据...")
    data_loader = TrajectoryDataLoader(
        batch_size=64,
        shuffle=False,
        num_workers=4,
        test_only=False,
        random_state=42,
    )

    _, _, test_loader = data_loader.get_dataloaders()
    
    # 进行综合评估
    evaluation_results = evaluate_streaming(model, test_loader, device, detailed_analysis=False)
    
    return evaluation_results


def main():
    """主函数"""
    print("🔍 流式MultiRocket模型评估系统")
    print("=" * 60)
    
    # 检查文件
    checkpoint_path = "checkpoints/best_streaming_model.pth"

    if not os.path.exists(checkpoint_path):
        print(f"❌ 模型文件不存在: {checkpoint_path}")
        print("请先运行训练脚本生成模型")
        return
    
    result = comprehensive_model_evaluation(checkpoint_path)
    # export_results(result, DATA_ROOT, DATA_ROOT)


if __name__ == "__main__":
    main()