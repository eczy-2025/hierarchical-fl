import torch
import copy
import threading
from src.modules.model import SimpleCNN

class CloudModelManager:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.lock = threading.Lock()
        
        # 1. 初始化全局模型 (随机权重)
        self.logger.info("Initializing global model...")
        self.global_model = SimpleCNN()
        self.device = torch.device(config['training']['device'])
        self.global_model.to(self.device)
        
        self.received_edge_updates = []
        
        # 【新增】全局轮次计数器
        self.global_round = 0

    def get_global_weights(self):
        """获取当前全局模型参数 (CPU)"""
        with self.lock:
            # 【修改】返回权重 + 当前轮次 ID
            return {
                "weights": {k: v.cpu() for k, v in self.global_model.state_dict().items()},
                "global_round": self.global_round
            }

    def add_edge_update(self, edge_data):
        """接收 Edge 上传"""
        with self.lock:
            self.received_edge_updates.append(edge_data)

    def check_aggregation_condition(self):
        """判断是否收到足够多的 Edge 更新"""
        # 简单策略：每收到 2 个 Edge 更新就聚合 (可根据实际拓扑修改)
        with self.lock:
            return len(self.received_edge_updates) >= 2

    def aggregate(self):
        """执行全局聚合"""
        with self.lock:
            if not self.received_edge_updates: return False
            
            updates = self.received_edge_updates
            total_samples = sum([u['samples'] for u in updates])
            
            # FedAvg: 加权平均
            base_weights = copy.deepcopy(updates[0]['weights'])
            for key in base_weights.keys():
                base_weights[key] = torch.zeros_like(base_weights[key], dtype=torch.float)
                
            for u in updates:
                factor = u['samples'] / total_samples
                for key in base_weights.keys():
                    base_weights[key] += u['weights'][key] * factor
            
            # 更新全局模型
            self.global_model.load_state_dict(base_weights)
            self.received_edge_updates = [] # 清空
            
            # 【关键】聚合完成后，轮次 +1
            self.global_round += 1
            return True

    def evaluate(self, test_loader):
        """核心功能：在测试集上评估全局模型"""
        self.global_model.eval()
        test_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.global_model(data)
                
                # 计算 Loss
                test_loss += torch.nn.functional.nll_loss(output, target, reduction='sum').item()
                
                # 计算 Accuracy
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += len(target)

        test_loss /= total
        accuracy = 100. * correct / total
        
        return test_loss, accuracy
