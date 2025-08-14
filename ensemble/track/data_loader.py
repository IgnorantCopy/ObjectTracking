"""
数据加载器模块
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, Any

from ensemble.track.data.preprocessor import TrajectoryPreprocessor


class TrajectoryDataset(Dataset):
    """轨迹数据集类"""
    def __init__(self, sequences: np.ndarray, labels: np.ndarray, batch_ids: np.ndarray, num_points: np.ndarray):
        """
        初始化数据集
        
        Args:
            sequences: 序列数据 (N, seq_len, features)
            labels: 标签数据 (N,)
            batch_ids: 批次ID (N,) 可选
        """
        self.sequences = torch.from_numpy(sequences).float()
        self.labels = torch.from_numpy(labels).long()
        self.batch_ids = batch_ids
        self.num_points = num_points
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return {
            'sequences': self.sequences[idx],
            'labels': self.labels[idx],
            'batch_ids': self.batch_ids[idx],
            'num_points': self.num_points[idx]
        }


class TrajectoryDataLoader:
    """轨迹数据加载器管理类"""
    
    def __init__(self, 
                 batch_size: int = 32,
                 train_split: float = 0.8,
                 val_split: float = 0.1,
                 test_split: float = 0.1,
                 shuffle: bool = True,
                 num_workers: int = 4,
                 normalize: bool = True,
                 test_only: bool = False,
                 random_state: int = 42):
        """
        初始化数据加载器
        
        Args:
            batch_size: 批次大小
            train_split: 训练集比例
            val_split: 验证集比例  
            test_split: 测试集比例
            shuffle: 是否打乱数据
            num_workers: 数据加载线程数
            normalize: 是否标准化特征
            random_state: 随机种子
        """
        self.batch_size = batch_size
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.normalize = normalize
        self.test_only = test_only
        self.random_state = random_state
        
        # 数据和数据加载器
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        
        self.preprocessor = TrajectoryPreprocessor(test=test_only)

        # 数据统计信息
        self.data_info = {}
        
        # 加载和准备数据
        self._load_and_prepare_data()

    def _load_and_prepare_data(self):
        """加载和准备数据"""
        # 加载数据
        data = self.preprocessor.process_all_trajectories()
        sequences = data['sequences']  # (N, seq_len, features)
        labels = data['labels']        # (N,)
        batch_ids = data['batch_ids']  # (N,)
        num_points = data['num_points']  # (N,)
        
        # 记录数据信息
        self.data_info = {
            'total_samples': len(sequences),
            'seq_len': sequences.shape[1],
            'num_features': sequences.shape[2],
            'num_classes': len(np.unique(labels)),
            'class_distribution': dict(zip(*np.unique(labels, return_counts=True)))
        }
        
        print(f"数据加载完成:")
        print(f"  总样本数: {self.data_info['total_samples']}")
        print(f"  序列长度: {self.data_info['seq_len']}")
        print(f"  特征维度: {self.data_info['num_features']}")
        print(f"  类别数量: {self.data_info['num_classes']}")
        print(f"  类别分布: {self.data_info['class_distribution']}")
        
        # 数据分割
        if self.test_only or self.val_split == 0.0:
            print(f"  总样本: {len(sequences)}")

            self.test_dataset = TrajectoryDataset(sequences, labels, batch_ids, num_points)
            self.test_loader = DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                num_workers=self.num_workers,
                pin_memory=torch.cuda.is_available()
            )
        else:
            train_sequences, temp_sequences, train_labels, temp_labels, train_batch_ids, temp_batch_ids, \
                train_num_points, temp_num_points = train_test_split(
                sequences, labels, batch_ids, num_points,
                test_size=(self.val_split + self.test_split),
                random_state=self.random_state,
                stratify=labels
            )

            # 计算验证集和测试集的相对比例
            val_test_split = self.val_split / (self.val_split + self.test_split)

            val_sequences, test_sequences, val_labels, test_labels, val_batch_ids, test_batch_ids, \
                val_num_points, test_num_points = train_test_split(
                temp_sequences, temp_labels, temp_batch_ids, temp_num_points,
                test_size=(1 - val_test_split),
                random_state=self.random_state,
                stratify=temp_labels
            )

            print(f"\n数据分割:")
            print(f"  训练集: {len(train_sequences)} 样本")
            print(f"  验证集: {len(val_sequences)} 样本")
            print(f"  测试集: {len(test_sequences)} 样本")
        
            # 创建数据集
            self.train_dataset = TrajectoryDataset(train_sequences, train_labels, train_batch_ids, train_num_points)
            self.val_dataset = TrajectoryDataset(val_sequences, val_labels, val_batch_ids, val_num_points)
            self.test_dataset = TrajectoryDataset(test_sequences, test_labels, test_batch_ids, test_num_points)

            # 创建数据加载器
            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                num_workers=self.num_workers,
                pin_memory=torch.cuda.is_available()
            )

            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=torch.cuda.is_available()
            )

            self.test_loader = DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=torch.cuda.is_available()
            )

        print(f"  数据加载器创建完成")
    
    def get_dataloaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """获取数据加载器"""
        return self.train_loader, self.val_loader, self.test_loader
    
    def get_datasets(self) -> Tuple[TrajectoryDataset, TrajectoryDataset, TrajectoryDataset]:
        """获取数据集"""
        return self.train_dataset, self.val_dataset, self.test_dataset
    
    def get_data_info(self) -> Dict[str, Any]:
        """获取数据信息"""
        return self.data_info