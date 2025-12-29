import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

class DataSlicer:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.data_path = './data'
        
        # 预处理
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        # 加载完整数据集
        # download=True 确保首次运行时自动下载
        self.train_dataset = datasets.MNIST(
            self.data_path, train=True, download=True, transform=self.transform
        )
        self.test_dataset = datasets.MNIST(
            self.data_path, train=False, download=True, transform=self.transform
        )
        
        # 缓存切分结果 {partition_idx: [data_indices]}
        self.client_indices_map = {} 
        
        # 定义总切片数（模拟的总 Client 数量池）
        # 只要这个数字大于实际运行的 Client 总数即可
        self.total_partitions = 100 
        
        self._partition_data()

    def _partition_data(self):
        """核心逻辑：根据配置切分数据"""
        partition_type = self.config['dataset']['partition']
        
        self.logger.info(f"Partitioning data: {partition_type} (Total partitions: {self.total_partitions})")

        if partition_type == "iid":
            self._partition_iid()
        elif partition_type == "noniid":
            beta = self.config['dataset'].get('beta', 0.5)
            self._partition_dirichlet(beta)
        else:
            self.logger.error("Unknown partition type")

    def _partition_iid(self):
        """IID 划分：随机打乱后均分"""
        num_items = int(len(self.train_dataset) / self.total_partitions)
        all_idxs = np.arange(len(self.train_dataset))
        np.random.shuffle(all_idxs)
        
        for i in range(self.total_partitions):
            self.client_indices_map[i] = all_idxs[i*num_items : (i+1)*num_items]

    def _partition_dirichlet(self, beta):
        """
        Non-IID 划分：Dirichlet 分布
        """
        min_size = 0
        y_train = np.array(self.train_dataset.targets)
        K = 10 # 类别数
        N = y_train.shape[0]
        
        while min_size < 10:
            idx_batch = [[] for _ in range(self.total_partitions)]
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(beta, self.total_partitions))
                
                proportions = np.array([p * (len(idx_j) < N / self.total_partitions) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for i in range(self.total_partitions):
            self.client_indices_map[i] = idx_batch[i]

    def get_client_dataloader(self, client_id):
        """
        获取指定 Client 的 DataLoader
        【修正方案 B】优化 ID 映射逻辑，避免不同 Edge 下的 ID 冲突
        """
        
        # 映射逻辑：
        # 如果 ID 是 3 位数 (如 101, 201)，我们利用百位作为区分 Edge 的依据
        # 101 -> Edge 1, Local 1
        # 201 -> Edge 2, Local 1
        # 公式: (Edge_ID * 20 + Local_ID) % total_partitions
        # 这样 101 映射到 21，201 映射到 41，数据就完全错开了
        
        if client_id >= 100:
            edge_id = client_id // 100
            local_id = client_id % 100
            # 假设每个 Edge 下最多 20 个节点，这样可以保证区间不重叠
            mapped_idx = (edge_id * 20 + local_id) % self.total_partitions
        else:
            # 对于普通 ID (1, 2, 3...) 直接使用
            mapped_idx = client_id % self.total_partitions

        self.logger.debug(f"Client {client_id} mapped to partition index: {mapped_idx}")

        if mapped_idx not in self.client_indices_map:
            mapped_idx = 0 # Fallback
            
        indices = self.client_indices_map[mapped_idx]
        subset = Subset(self.train_dataset, indices)
        
        return DataLoader(
            subset, 
            batch_size=self.config['training']['batch_size'], 
            shuffle=True
        )

    def get_test_dataloader(self):
        """获取全局测试集 (用于 Cloud 评估)"""
        return DataLoader(
            self.test_dataset, 
            batch_size=1000, 
            shuffle=False
        )