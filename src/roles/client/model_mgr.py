import torch
import torch.nn as nn
import torch.optim as optim
from src.modules.model import SimpleCNN

class ClientModelManager:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.device = torch.device(config['training']['device'])
        
        # 1. 内部初始化模型 (工厂模式)
        # 这样 manager.py 就不需要手动实例化 model 再传进来了
        self.model = SimpleCNN(num_classes=config['dataset']['num_classes']).to(self.device)
        
        # 2. 定义损失函数
        self.criterion = nn.CrossEntropyLoss()

    def load_weights(self, weights):
        """加载从 Edge 下发的全局模型权重"""
        self.model.load_state_dict(weights)
        # 再次确保模型在正确的设备上
        self.model.to(self.device)

    def get_weights(self):
        """获取当前本地模型的权重"""
        # 必须转为 CPU，否则序列化传输时会报错
        return {k: v.cpu() for k, v in self.model.state_dict().items()}

    def train(self, train_loader, assigned_epochs=None):
        """
        执行本地训练
        :param train_loader: DataLoader 对象
        :param assigned_epochs: Edge 动态下发的自适应训练轮次
        :return: (weights, avg_loss, sample_count)
        """
        self.model.train()

        # 【新增】如果有 Edge 下发的定制 Epoch，则使用定制值；否则回退到默认配置
        if assigned_epochs is not None:
            epochs = assigned_epochs
            self.logger.info(f"Ada t&d-aware: Training for {epochs} epochs based on Edge instruction.")
        else:
            epochs = self.config['training']['local_epochs']
            self.logger.info(f"Pre-training/Fallback: Training for fixed {epochs} epochs.")

        lr = self.config['training']['learning_rate']
        momentum = self.config['training']['momentum']
        
        # 每次训练开始时重新初始化优化器 (因为模型参数可能被 load_weights 更新过)
        optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)
        
        total_loss = 0.0
        sample_count = 0
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            total = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item() * data.size(0) # 累积绝对 Loss
                total += data.size(0)
            
            # 记录最后一个 Epoch 的统计信息
            if epoch == epochs - 1:
                total_loss = epoch_loss
                sample_count = total
        
        avg_loss = total_loss / sample_count if sample_count > 0 else 0.0
        
        # 返回更新后的权重、Loss和样本数
        return self.get_weights(), avg_loss, sample_count
