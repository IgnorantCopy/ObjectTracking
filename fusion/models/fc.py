import torch.nn as nn


class FC(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=1024, dropout=0.5):
        super().__init__()
        self.layers = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        return self.layers(x)