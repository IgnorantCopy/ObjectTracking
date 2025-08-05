"""
流式MultiRocket模型训练脚本
"""
import sys
from pathlib import Path

# 添加项目根目录到路径
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent  # 回到项目根目录
sys.path.append(str(project_root))

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
from tqdm import tqdm

from ensemble.src.training.streaming_multi_rocket import StreamingMultiRocketClassifier, StreamingTrainer
from ensemble.src.training.data_loader import TrajectoryDataLoader
from ensemble.src.training.streaming_config import StreamingConfig


def train_and_evaluate_streaming_model(config: StreamingConfig):
    """训练和评估流式模型"""

    print("Starting Streaming MultiRocket Model Training")
    print("=" * 60)

    # 设备检查
    if config.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config.device)
    print(f"Device: {device}")

    # 创建保存目录
    save_dir = Path(config.save_dir)
    save_dir.mkdir(exist_ok=True)

    # 数据加载
    print("\\nLoading data...")
    data_loader = TrajectoryDataLoader(
        batch_size=config.batch_size,
        train_split=config.train_split,
        val_split=config.val_split,
        test_split=config.test_split,
        shuffle=config.shuffle,
        num_workers=0,
        random_state=config.seed
    )

    data_info = data_loader.get_data_info()
    print("Data loaded successfully:")
    for key, value in data_info.items():
        print(f"  {key}: {value}")


    # 模型创建
    print("\\nCreating streaming model...")
    model = StreamingMultiRocketClassifier(
        c_in=data_info['num_features'],
        c_out=data_info['num_classes'],
        max_seq_len=data_info['seq_len'],
        # min_seq_len=config.min_seq_len,
        num_features=config.num_features,
        dropout=config.dropout,
        confidence_threshold=config.confidence_threshold
    ).to(device)

    print("Model created successfully")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # 训练设置
    trainer = StreamingTrainer(model, device)
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    train_loader, val_loader, test_loader = data_loader.get_dataloaders()

    # 训练历史
    log_dir = os.path.join(save_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    best_val_acc = 0.0
    patience_counter = 0

    print(f"\\nStarting training ({config.epochs} epochs)...")
    print("=" * 60)

    # 训练循环
    for epoch in range(config.epochs):
        print(f"\\nEpoch {epoch + 1}/{config.epochs}")

        # 训练阶段
        model.train()
        train_metrics = {
            'loss': 0.,
            'begin_time': 0.,
            'corrects': torch.zeros((data_info['num_classes'],), device=device),
            'totals': torch.zeros((data_info['num_classes'],), device=device)
        }

        for batch in tqdm(train_loader, desc="Training"):
            sequences = batch['sequences'].to(device)
            labels = batch['labels'].to(device)
            sequences = sequences.transpose(1, 2)

            metrics = trainer.train_step(sequences, labels, optimizer)
            for key, value in metrics.items():
                train_metrics[key] += value

        # 计算训练指标
        train_loss = train_metrics['loss'] / train_metrics['totals'].sum()
        train_avg_acc = train_metrics['corrects'].sum() / train_metrics['totals'].sum()
        train_avg_begin_time = train_metrics['begin_time'] / train_metrics['totals'].sum()
        train_metrics['totals'][train_metrics['totals'] == 0] = 1  # 防止除零错误
        accuracies = train_metrics['corrects'] / train_metrics['totals']
        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("train/avg_acc", train_avg_acc, epoch)
        writer.add_scalar("train/avg_begin_time", train_avg_begin_time, epoch)
        for i in range(data_info['num_classes']):
            writer.add_scalar(f"train/acc_{i}", accuracies[i], epoch)

        # 验证阶段
        print("Validating...")
        model.eval()
        with torch.no_grad():
            val_metrics = {
                'loss': 0.,
                'begin_time': 0.,
                'corrects': torch.zeros((data_info['num_classes'],), device=device),
                'totals': torch.zeros((data_info['num_classes'],), device=device)
            }
            for batch in tqdm(val_loader, desc="Validating"):
                sequences = batch['sequences'].to(device)
                labels = batch['labels'].to(device)
                sequences = sequences.transpose(1, 2)

                metrics = trainer.evaluate_step(sequences, labels)
                for key, value in metrics.items():
                    val_metrics[key] += value

        val_loss = val_metrics['loss'] / val_metrics['totals'].sum()
        val_avg_acc = val_metrics['corrects'].sum() / val_metrics['totals'].sum()
        val_avg_begin_time = val_metrics['begin_time'] / val_metrics['totals'].sum()
        val_metrics['totals'][val_metrics['totals'] == 0] = 1  # 防止除零错误
        accuracies = val_metrics['corrects'] / val_metrics['totals']
        writer.add_scalar("val/loss", val_loss, epoch)
        writer.add_scalar("val/avg_acc", val_avg_acc, epoch)
        writer.add_scalar("val/avg_begin_time", val_avg_begin_time, epoch)
        for i in range(data_info['num_classes']):
            writer.add_scalar(f"val/acc_{i}", accuracies[i], epoch)

        # 更新学习率
        writer.add_scalar("learning_rate", optimizer.param_groups[0]['lr'], epoch)
        scheduler.step(val_avg_acc)

        # 打印结果
        print(f"Results:")
        print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_avg_acc:.4f}")
        print(f"  Valid - Loss: {val_loss:.4f}, Acc: {val_avg_acc:.4f}")

        # 保存最佳模型
        if val_avg_acc > best_val_acc:
            best_val_acc = val_avg_acc
            patience_counter = 0

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_avg_acc,
                'config': config,
                'data_info': data_info,
                'supported_lengths': model.supported_lengths
            }, os.path.join(save_dir, 'best_streaming_model.pth'))

            print(f"  Best model saved (acc: {val_avg_acc:.4f})")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{config.early_stopping_patience})")

        # 早停检查
        if patience_counter >= config.early_stopping_patience:
            print("Early stopping triggered!")
            break

    # 最终测试
    print("\\n" + "=" * 60)
    print("Final testing...")

    # 加载最佳模型
    best_model_path = os.path.join(save_dir, 'best_streaming_model.pth')
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Best model loaded for testing")

    with torch.no_grad():
        test_metrics = {
            'loss': 0.,
            'begin_time': 0.,
            'corrects': torch.zeros((data_info['num_classes'],), device=device),
            'totals': torch.zeros((data_info['num_classes'],), device=device)
        }
        for batch in tqdm(test_loader, desc="Testing"):
            sequences = batch['sequences'].to(device)
            labels = batch['labels'].to(device)
            sequences = sequences.transpose(1, 2)

            metrics = trainer.evaluate_step(sequences, labels)
            for key, value in metrics.items():
                test_metrics[key] += value

    test_avg_acc = test_metrics['corrects'].sum() / test_metrics['totals'].sum()
    test_avg_begin_time = test_metrics['begin_time'] / test_metrics['totals'].sum()
    test_metrics['totals'][test_metrics['totals'] == 0] = 1  # 防止除零错误
    print("\\nFinal Test Results:")
    print(f"  Accuracy: {test_avg_acc:.4f}")
    print(f"  Avg Begin Time: {test_avg_begin_time:.1f}")

    # save scaler
    scaler_path = os.path.join(save_dir, 'data_scaler.pth')
    data_loader.save_scaler(scaler_path)

    print("\\nTraining completed!")
    print("Files saved:")
    print("  - checkpoints/best_streaming_model.pth")


def main():
    """主函数"""
    print("Streaming MultiRocket Model Training System")
    print("=" * 60)
    
    # 配置
    config = StreamingConfig()
    
    print("Training Configuration:")
    print(f"  Epochs: {config.epochs}")
    print(f"  Batch Size: {config.batch_size}")
    print(f"  Learning Rate: {config.learning_rate}")
    print(f"  Confidence Threshold: {config.confidence_threshold}")
    
    # 训练和评估
    train_and_evaluate_streaming_model(config)


if __name__ == "__main__":
    main()