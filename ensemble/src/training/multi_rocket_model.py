"""
使用HydraMultiRocketBackbonePlus的完整模型包装器
"""
import torch.nn as nn
from tsai.models.MultiRocketPlus import MultiRocketBackbonePlus

class MultiRocketPlusClassifier(nn.Module):
    """MultiRocketBackbonePlus + 分类头的完整模型"""
    
    def __init__(self, c_in, c_out, seq_len, num_features, dropout=0.2, **kwargs):
        super().__init__()
        
        self.backbone = MultiRocketBackbonePlus(
            c_in=c_in,
            seq_len=seq_len,
            num_features=num_features,
            **kwargs
        )
        
        backbone_out_dim = self.backbone.num_features

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.BatchNorm1d(backbone_out_dim),
            nn.Dropout(dropout),
            nn.Linear(backbone_out_dim, c_out),
        )
        
    def forward(self, x):
        features = self.backbone(x)
        output = self.classifier(features)
        return output