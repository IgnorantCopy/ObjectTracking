"""
主执行脚本 - 数据预处理
"""
import os
import sys
import argparse

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ensemble.src.data_preprocessing.preprocessor import TrajectoryPreprocessor
from ensemble.src.utils.config import DATA_ROOT, SEQ_LEN, NUM_CLASSES


def main():
    parser = argparse.ArgumentParser(description='轨迹数据预处理')
    parser.add_argument('--data_root', type=str, default=DATA_ROOT, 
                        help='数据根目录')
    parser.add_argument('--seq_len', type=int, default=SEQ_LEN,
                        help='序列长度')
    parser.add_argument('--output', type=str, default='processed_data.npz',
                        help='输出文件路径')
    parser.add_argument('--verbose', action='store_true',
                        help='详细输出')
    
    args = parser.parse_args()
    
    # 检查数据目录
    if not os.path.exists(args.data_root):
        print(f"错误: 数据目录不存在 {args.data_root}")
        return
    
    point_dir = os.path.join(args.data_root, "点迹")
    track_dir = os.path.join(args.data_root, "航迹")
    
    if not os.path.exists(point_dir):
        print(f"错误: 点迹目录不存在 {point_dir}")
        return
    
    if not os.path.exists(track_dir):
        print(f"错误: 航迹目录不存在 {track_dir}")
        return
    
    # 初始化预处理器
    preprocessor = TrajectoryPreprocessor(
        data_root=args.data_root,
        seq_len=args.seq_len,
        num_classes=NUM_CLASSES,
        verbose=args.verbose
    )
    
    print("开始处理轨迹数据...")
    print(f"数据根目录: {args.data_root}")
    print(f"序列长度: {args.seq_len}")
    print(f"类别数量: {NUM_CLASSES}")
    
    # 处理所有轨迹
    results = preprocessor.process_all_trajectories()
    
    if results['success_count'] == 0:
        print("错误: 没有成功处理任何轨迹数据")
        return
    
    # 保存结果
    preprocessor.save_processed_data(args.output)
    
    # 输出统计信息
    print(f"\n处理完成!")
    print(f"成功处理: {results['success_count']} 个轨迹")
    print(f"处理失败: {results['fail_count']} 个轨迹")
    print(f"序列形状: {results['sequences'].shape}")
    print(f"标签分布: {dict(zip(*np.unique(results['labels'], return_counts=True)))}")
    print(f"数据已保存到: {args.output}")
    
    # 验证PyTorch兼容性
    try:
        X, y = preprocessor.get_pytorch_dataset()
        print(f"\nPyTorch张量形状:")
        print(f"X: {X.shape}")
        print(f"y: {y.shape}")
        print("数据格式验证通过!")
    except Exception as e:
        print(f"PyTorch格式转换失败: {e}")


if __name__ == "__main__":
    import numpy as np
    main()