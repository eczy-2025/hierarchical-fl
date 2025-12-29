import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        # 卷积层 1: 1通道输入(灰度图) -> 10通道输出
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        # 卷积层 2: 10通道输入 -> 20通道输出
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        
        # 全连接层
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        # Conv1 -> Pool -> Relu
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # Conv2 -> Drop -> Pool -> Relu
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        
        # Flatten (展平)
        x = x.view(-1, 320)
        
        # FC1 -> Relu
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        
        # FC2
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
