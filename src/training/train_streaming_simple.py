"""
流式 MultiRocket 模型训练脚本
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

from src.training.streaming_multi_rocket import StreamingMultiRocketClassifier, StreamingTrainer
from src.training.data_loader import TrajectoryDataLoader
from src.training.streaming_config import StreamingConfig


def get_data_loader(config: StreamingConfig):
    """获取数据加载器"""
    # 数据加载
    print("\nLoading data...")
    return TrajectoryDataLoader(
        batch_size=config.batch_size,
        train_split=config.train_split,
        val_split=config.val_split,
        test_split=config.test_split,
        shuffle=config.shuffle,
        num_workers=0,
        random_state=config.seed
    )


def train_and_evaluate_streaming_model(config: StreamingConfig, data_loader: TrajectoryDataLoader):
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

    data_info = data_loader.get_data_info()
    print("Data loaded successfully:")
    for key, value in data_info.items():
        print(f"  {key}: {value}")


    # 模型创建
    print("\nCreating streaming model...")
    num_classes = data_info['num_classes']
    seq_len = data_info['seq_len']

    model = StreamingMultiRocketClassifier(
        c_in=data_info['num_features'],
        c_out=num_classes,
        max_seq_len=seq_len,
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
    optimizers = []
    schedulers = []
    for i in range(seq_len):
        optimizer = optim.Adam(model.models[i].parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)
        optimizers.append(optimizer)
        schedulers.append(scheduler)

    train_loader, val_loader, test_loader = data_loader.get_dataloaders()

    # 训练历史
    log_dir = os.path.join(save_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    best_model_state_dicts = [m.state_dict() for m in model.models]
    best_val_accuracies = [0. for _ in range(seq_len)]
    best_val_acc = 0.
    patience_counter = 0

    print(f"\nStarting training ({config.epochs} epochs)...")
    print("=" * 60)

    # 训练循环
    for epoch in range(config.epochs):
        print(f"\nEpoch {epoch + 1}/{config.epochs}")

        # 训练阶段
        model.train()
        train_metrics = {
            'begin_time': 0.,
            'loss': torch.zeros((seq_len,), device=device),
            'corrects': torch.zeros((seq_len, num_classes), device=device),
            'totals': torch.zeros((seq_len, num_classes), device=device)
        }

        for batch in tqdm(train_loader, desc="Training"):
            sequences = batch['sequences'].to(device)
            labels = batch['labels'].to(device)
            sequences = sequences.transpose(1, 2)

            metrics = trainer.train_step(sequences, labels, optimizers)
            for key, value in metrics.items():
                if key in train_metrics:
                    train_metrics[key] += value

        # 计算训练指标
        train_losses = train_metrics['loss'] / train_metrics['totals'].sum(1)
        train_avg_accuracies = train_metrics['corrects'].sum(1) / train_metrics['totals'].sum(1)
        train_acc = train_metrics['corrects'].sum() / train_metrics['totals'].sum()
        train_avg_begin_time = train_metrics['begin_time'] / train_metrics['totals'].sum()
        train_metrics['totals'][train_metrics['totals'] == 0] = 1  # 防止除零错误
        accuracies = train_metrics['corrects'] / train_metrics['totals']

        writer.add_scalar("train/avg_acc", train_acc, epoch)
        writer.add_scalar(f"train/avg_begin_time", train_avg_begin_time, epoch)
        for i in range(seq_len):
            writer.add_scalar(f"model{i+1}/train/loss", train_losses[i], epoch)
            writer.add_scalar(f"model{i+1}/train/avg_acc", train_avg_accuracies[i], epoch)
            for j in range(num_classes):
                writer.add_scalar(f"model{i+1}/train/acc_{j}", accuracies[i][j], epoch)

        # 验证阶段
        model.eval()
        with torch.no_grad():
            val_metrics = {
                'begin_time': 0.,
                'loss': torch.zeros((seq_len,), device=device),
                'corrects': torch.zeros((seq_len, num_classes), device=device),
                'totals': torch.zeros((seq_len, num_classes), device=device)
            }
            for batch in tqdm(val_loader, desc="Validating"):
                sequences = batch['sequences'].to(device)
                labels = batch['labels'].to(device)
                sequences = sequences.transpose(1, 2)

                metrics = trainer.evaluate_step(sequences, labels)
                for key, value in metrics.items():
                    if key in val_metrics:
                        val_metrics[key] += value

        val_losses = val_metrics['loss'] / val_metrics['totals'].sum(1)
        val_avg_accuracies = val_metrics['corrects'].sum(1) / val_metrics['totals'].sum(1)
        val_acc = val_metrics['corrects'].sum() / val_metrics['totals'].sum()
        val_avg_begin_time = val_metrics['begin_time'] / val_metrics['totals'].sum()
        val_metrics['totals'][val_metrics['totals'] == 0] = 1  # 防止除零错误
        accuracies = val_metrics['corrects'] / val_metrics['totals']

        writer.add_scalar("val/avg_acc", val_acc, epoch)
        writer.add_scalar("val/avg_begin_time", val_avg_begin_time, epoch)
        for i in range(seq_len):
            writer.add_scalar(f"model{i+1}/val/loss", val_losses[i], epoch)
            writer.add_scalar(f"model{i+1}/val/avg_acc", val_avg_accuracies[i], epoch)
            for j in range(num_classes):
                writer.add_scalar(f"model{i+1}/val/acc_{j}", accuracies[i][j], epoch)

            # 更新学习率
            writer.add_scalar(f"model{i+1}/learning_rate", optimizers[i].param_groups[0]['lr'], epoch)
            schedulers[i].step(val_avg_accuracies[i])

        # 打印结果
        print(f"Train\n\tLoss: {train_losses.cpu().tolist()}\n\tAcc: {train_avg_accuracies.cpu().tolist()}")
        print(f"Valid\n\tLoss: {val_losses.cpu().tolist()}\n\tAcc: {val_avg_accuracies.cpu().tolist()}")

        # 保存最佳模型
        have_progress = False
        for i in range(seq_len):
            if val_avg_accuracies[i] > best_val_accuracies[i]:
                best_val_accuracies[i] = val_avg_accuracies[i]
                patience_counter = 0
                have_progress = True

                best_model_state_dicts[i] = model.models[i].state_dict()

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': best_model_state_dicts[i],
                    'optimizer_state_dict': optimizers[i].state_dict(),
                    'val_accuracy': val_avg_accuracies,
                    'config': config,
                    'data_info': data_info,
                }, os.path.join(save_dir, f'multi_rocket_model_{i+1}.pth'))

                print(f"Best multi_rocket_model_{i+1} saved (acc: {val_avg_accuracies[i]:.4f})")
        if not have_progress:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{config.early_stopping_patience})")

        # 早停检查
        best_val_acc = max(best_val_acc, val_acc)
        if patience_counter >= config.early_stopping_patience:
            print("Early stopping triggered!")
            break

    # 最终测试
    print("\\n" + "=" * 60)
    print("Final testing...")

    # 加载最佳模型
    for i in range(seq_len):
        model.models[i].load_state_dict(best_model_state_dicts[i])
    print("Best model loaded for testing")

    with torch.no_grad():
        test_metrics = {
            'begin_time': 0.,
            'loss': torch.zeros((seq_len,), device=device),
            'corrects': torch.zeros((seq_len, num_classes), device=device),
            'totals': torch.zeros((seq_len, num_classes), device=device)
        }
        for batch in tqdm(test_loader, desc="Testing"):
            sequences = batch['sequences'].to(device)
            labels = batch['labels'].to(device)
            sequences = sequences.transpose(1, 2)

            metrics = trainer.evaluate_step(sequences, labels)
            for key, value in metrics.items():
                if key in test_metrics:
                    test_metrics[key] += value

    test_avg_accuracies = test_metrics['corrects'].sum(1) / test_metrics['totals'].sum(1)
    test_avg_begin_time = test_metrics['begin_time'] / test_metrics['totals'].sum()
    test_metrics['totals'][test_metrics['totals'] == 0] = 1  # 防止除零错误
    print("\\nFinal Test Results:")
    print(f"  Accuracy: {test_avg_accuracies.cpu().tolist()}")
    print(f"  Avg Begin Time: {test_avg_begin_time:.1f}")

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'val_accuracy': best_val_acc,
        'config': config,
        'data_info': data_info,
    }, os.path.join(save_dir, f'best_streaming_model.pth'))

    return best_val_acc


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

    data_loader = get_data_loader(config)

    # 训练和评估
    train_and_evaluate_streaming_model(config, data_loader)



if __name__ == "__main__":
    main()