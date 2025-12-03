import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Feature Extraction (The "Eye")
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        
        # Classification (The "Brain")
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(4096, 10)  # 16*16*16 -> 10

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.flatten(x)
        x = self.fc1(x)
        return x