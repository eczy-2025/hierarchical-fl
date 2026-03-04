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

        # 全局轮次计数器
        self.global_round = 0

        # 【新增】FedDoMA 动态触发与加权相关的状态变量
        self.current_distance_threshold = 0.1  # 初始目标距离
        self.max_queue_size = 2  # 兜底最大队列长度 (可配)

    def get_global_weights(self):
        """获取当前全局模型参数 (CPU)"""
        with self.lock:
            return {
                "weights": {k: v.cpu() for k, v in self.global_model.state_dict().items()},
                "global_round": self.global_round
            }

    def add_edge_update(self, edge_data):
        """接收 Edge 上传"""
        with self.lock:
            self.received_edge_updates.append(edge_data)

    # =====================================================================
    # 核心改造 1: 动态异步触发阀门 (Dynamic Trigger)
    # =====================================================================
    def check_aggregation_condition(self):
        """
        判断是否满足 FedDoMA 聚合条件
        返回: (is_ready: bool, reason: str)
        """
        with self.lock:
            queue_size = len(self.received_edge_updates)
            if queue_size == 0:
                return False, ""

            # 阀门 A：数量触发 (兜底机制，防止无限等待)
            if queue_size >= self.max_queue_size:
                return True, f"Max Queue Size Reached ({queue_size}/{self.max_queue_size})"

            # 阀门 B：距离饱和触发 (预测更新幅度是否达到目标)
            w_local = self.received_edge_updates[0]['weights']
            w_global = self.global_model.state_dict()
            cur_distance = self._compute_L2norm(w_local, w_global)

            target_distance = 0.9 * self.current_distance_threshold

            if abs(cur_distance - target_distance) < 0.01:
                return True, f"Target Distance Reached (Delta: {abs(cur_distance - target_distance):.4f} < 0.01)"

            return False, ""

    # =====================================================================
    # 核心改造 2: 多元异步聚合算法 (MultiAsyncAgg)
    # =====================================================================
    def aggregate(self):
        """执行 FedDoMA 聚合"""
        with self.lock:
            if not self.received_edge_updates: return False

            updates = self.received_edge_updates
            num_updates = len(updates)
            total_samples = sum([u['samples'] for u in updates])

            w_global = self.global_model.state_dict()

            # 从配置读取超参数 (如果没有则给默认值)
            stale_ratio = self.config.get('training', {}).get('stale_ratio', 0.4)
            hetero_ratio = self.config.get('training', {}).get('hetero_ratio', 0.6)

            alpha_list = []  # 陈旧度权重列表
            sim_list = []  # 相似度权重列表
            theta_list = []  # 最终综合权重列表

            # 1. 计算 Alpha 和 Similarity
            for u in updates:
                # 计算陈旧度 Staleness = 当前全局轮次 - Edge训练时基于的轮次
                edge_version = u.get('version', self.global_round)
                staleness = max(0, self.global_round - edge_version)
                alpha = pow(1 + staleness, -0.6)
                alpha_list.append(alpha)

                # 计算 CKA 相似度
                w_local = u['weights']
                sim = self._compute_cosine_similarity(w_global, w_local)
                sim_list.append(sim)

            # 2. 计算综合权重 Theta_j
            for i in range(num_updates):
                p_j = updates[i]['samples'] / total_samples
                theta = p_j * (stale_ratio * alpha_list[i] + hetero_ratio * sim_list[i])
                theta_list.append(theta)
                self.logger.info(
                    f"[FedDoMA] Update {i} -> alpha: {alpha_list[i]:.4f}, sim: {sim_list[i]:.4f}, theta: {theta:.4f}")

            # 3. 执行加权聚合
            averaged_params = {}
            for k in w_global.keys():
                for i in range(num_updates):
                    if i == 0:
                        averaged_params[k] = updates[i]['weights'][k] * theta_list[i]
                    else:
                        averaged_params[k] += updates[i]['weights'][k] * theta_list[i]

            # 4. 防偏移修正项 (1 - Phi) * W_t
            phi = sum(theta_list)
            for k in w_global.keys():
                averaged_params[k] = w_global[k] * (1 - phi) + averaged_params[k]

            # 更新距离阈值，供下一轮触发器使用
            new_distance = self._compute_L2norm(averaged_params, w_global)
            self.current_distance_threshold = new_distance if new_distance > 0.0 else 0.1

            # 覆盖全局模型
            self.global_model.load_state_dict(averaged_params)
            self.received_edge_updates = []  # 清空队列
            self.global_round += 1
            return True

    # =====================================================================
    # 数学工具函数
    # =====================================================================
    def _compute_cosine_similarity(self, w1, w2):
        """计算两个模型的余弦相似度 (平替 CKA)"""
        dot_product = 0.0
        norm_w1 = 0.0
        norm_w2 = 0.0
        for k in w1.keys():
            vec1 = w1[k].view(-1)
            vec2 = w2[k].view(-1)
            dot_product += torch.sum(vec1 * vec2).item()
            norm_w1 += torch.sum(vec1 * vec1).item()
            norm_w2 += torch.sum(vec2 * vec2).item()

        if norm_w1 == 0 or norm_w2 == 0: return 0.0
        sim = dot_product / ((norm_w1 ** 0.5) * (norm_w2 ** 0.5))
        return max(0.0, sim)

    def _compute_L2norm(self, w1, w2):
        """计算两个模型参数差值的 L2 范数平方"""
        diff_norm = 0.0
        for k in w1.keys():
            diff_norm += (w1[k] - w2[k]).norm(2).item() ** 2
        return diff_norm

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
                test_loss += torch.nn.functional.nll_loss(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += len(target)

        test_loss /= total
        accuracy = 100. * correct / total
        return test_loss, accuracy