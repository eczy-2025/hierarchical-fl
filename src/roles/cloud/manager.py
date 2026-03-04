import time
from src.modules.communication import FLCommunication
from src.modules.data_loader import DataSlicer
from src.roles.cloud.model_mgr import CloudModelManager

class CloudManager:
    def __init__(self, config, port, logger):
        self.config = config
        self.port = port
        self.logger = logger
        
        self.comm = FLCommunication(port=port, logger=logger)
        self.model_mgr = CloudModelManager(config, logger)
        
        # 准备测试数据 (用于评估)
        self.logger.info("Loading Test Dataset...")
        slicer = DataSlicer(config, logger)
        self.test_loader = slicer.get_test_dataloader()
        
        self._register_routes()

    def _register_routes(self):
        # 接口 1: 接收 Edge 上传
        def handle_upload():
            from flask import request
            try:
                data = self.comm.deserialize_request(request)
                self.model_mgr.add_edge_update(data)
                self.logger.info(f"Received update from Edge {data['id']}")
                return "ACK"
            except Exception as e:
                self.logger.error(f"Upload error: {e}")
                return "ERR", 500

        # 接口 2: Edge 拉取全局模型
        def handle_download():
            weights = self.model_mgr.get_global_weights()
            return self.comm.pickle_response(weights)

        self.comm.add_endpoint('/upload_edge_model', 'upload', handle_upload, methods=['POST'])
        self.comm.add_endpoint('/get_global_model', 'download', handle_download, methods=['GET'])

    def start(self):
        # 1. 启动 Server (非阻塞)
        self.comm.start_server_thread()
        self.logger.info(f"Cloud Server started on port {self.port}")
        
        # 2. 初始评估 (Round 0)
        loss, acc = self.model_mgr.evaluate(self.test_loader)
        self.logger.info(f"Initial Global Model - Test Loss: {loss:.4f}, Accuracy: {acc:.2f}%")
        
        # 3. 主循环
        while True:
            # =================================================================
            # 【修改】检查是否满足 FedDoMA 的混合异步触发条件
            # 这里的 check_aggregation_condition 将返回两个值：(是否触发, 触发原因)
            # =================================================================
            is_ready, reason = self.model_mgr.check_aggregation_condition()

            if is_ready:
                # 打印出极具成就感的 FedDoMA 动态触发日志！
                self.logger.info(f"⚡ FedDoMA Triggered! Reason: {reason}")
                self.logger.info("Aggregating Edge models...")

                if self.model_mgr.aggregate():
                    # 聚合后立即评估
                    loss, acc = self.model_mgr.evaluate(self.test_loader)
                    round_id = self.model_mgr.global_round
                    self.logger.info(f"=== Global Round {round_id} Result ===")
                    self.logger.info(f"    Test Loss: {loss:.4f}")
                    self.logger.info(f"    Accuracy : {acc:.2f}%")
                    # (未来可以在这里把结果写入 CSV)
            
            time.sleep(2)
