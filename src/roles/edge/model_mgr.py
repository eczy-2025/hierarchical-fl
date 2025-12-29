import torch
import copy
import threading

class EdgeModelManager:
    def __init__(self, config, logger, expected_clients):
        self.config = config
        self.logger = logger
        self.lock = threading.Lock() 
        
        # 存储区
        self.received_updates = []
        self.latest_global_weights = None
        
        # 版本控制
        self.model_version = 0
        
        # 聚合阈值 (初始值，后续会被 reset_threshold 动态覆盖)
        self.threshold = expected_clients
        
        # 记录上一次同步的云端轮次，防止重复更新
        self.last_cloud_round = -1
        
    def reset_threshold(self, num_participants):
        """
        【新增】根据实际报名的客户端数量，动态重置聚合阈值
        在招募窗口结束后调用。
        """
        with self.lock:
            # 至少为 1，防止除零错误或死锁
            self.threshold = max(1, num_participants)
            # 开启新一轮前，清空之前的残留更新 (如果有)

    def load_global_weights(self, payload):
        """
        更新本地缓存的全局模型
        :param payload: dict {'weights': ..., 'global_round': ...} 
                        或者仅包含 {'weights': ...} (本地聚合情况)
        :return: Boolean (是否进行了更新)
        """
        # 情况 A: Edge 本地聚合产生的更新（格式通常只包含 weights），总是更新
        if 'global_round' not in payload:
            with self.lock:
                self.latest_global_weights = payload['weights']
                self.model_version += 1
            return True

        # 情况 B: 来自 Cloud 的更新，需要检查轮次
        cloud_round = payload['global_round']
        weights = payload['weights']
        
        with self.lock:
            # 只有当云端轮次比当前大时，才更新
            if cloud_round > self.last_cloud_round:
                self.latest_global_weights = weights
                self.last_cloud_round = cloud_round
                self.model_version += 1
                return True
            else:
                # 轮次没变，忽略
                return False

    def get_latest_model(self):
        """供客户端下载"""
        with self.lock:
            if self.latest_global_weights is None:
                return None
            
            # 返回带版本号的字典
            return {
                "weights": self.latest_global_weights,
                "version": self.model_version
            }

    def add_client_update(self, client_data):
        """接收客户端上传"""
        with self.lock:
            self.received_updates.append(client_data)

    def check_aggregation_condition(self):
        """判断是否满足聚合条件"""
        with self.lock:
            # 比较当前的收集量与动态设置的阈值
            return len(self.received_updates) >= self.threshold

    def aggregate(self):
        """核心算法：执行 FedAvg"""
        with self.lock:
            if not self.received_updates:
                return None
            
            updates = self.received_updates
            num_updates = len(updates)
            
            # FedAvg
            total_samples = sum([u['samples'] for u in updates])
            aggregated_weights = copy.deepcopy(updates[0]['weights'])
            
            for key in aggregated_weights.keys():
                aggregated_weights[key] = torch.zeros_like(aggregated_weights[key], dtype=torch.float)
            
            for u in updates:
                factor = u['samples'] / total_samples
                for key in aggregated_weights.keys():
                    aggregated_weights[key] += u['weights'][key] * factor
            
            # 清空队列
            self.received_updates = []
            
            return {
                "weights": aggregated_weights,
                "samples": total_samples,
                "count": num_updates
            }