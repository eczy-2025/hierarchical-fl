import time
import random
from src.modules.communication import FLCommunication
from src.modules.data_loader import DataSlicer
from src.roles.client.model_mgr import ClientModelManager

class ClientManager:
    def __init__(self, config, client_id, edge_addr, edge_port, logger):
        self.config = config
        self.id = client_id
        self.logger = logger
        self.edge_url = f"http://{edge_addr}:{edge_port}"
        
        self.comm = FLCommunication(logger=logger)
        self.data_slicer = DataSlicer(config, logger)
        self.model_mgr = ClientModelManager(config, logger)
        
        # 预加载数据
        self.train_loader = self.data_slicer.get_client_dataloader(client_id)
        
        # 【新增】读取在线率配置 (默认为 1.0 即 100% 在线)
        self.online_rate = config['training'].get('online_rate', 1.0)

    def start(self):
        self.logger.info(f"Client {self.id} started. Connecting to Edge: {self.edge_url}")
        self.logger.info(f"Strategy: Online Rate = {self.online_rate}")
        
        # 本地版本记录，初始为 -1
        current_version = -1
        
        while True:
            # ================= PHASE 1: 等待 Edge 通知 & 报名 (Poll, Random Check, Join) =================
            latest_weights = None
            
            # 进入轮询等待状态
            while True:
                try:
                    # 1. 轮询 Edge 获取最新模型信息
                    response = self.comm.get_data(f"{self.edge_url}/get_latest_model")
                    
                    if response is None:
                        time.sleep(2)
                        continue
                        
                    remote_version = response['version']
                    
                    # 发现新版本 (Edge 已经开启了新的一轮)
                    if remote_version > current_version:
                        
                        # 2. 【新增】模拟设备随机掉线
                        # 掷骰子决定本轮是否参与
                        if random.random() > self.online_rate:
                            self.logger.info(f"Decided to be OFFLINE. Sleeping for 10s...")
                            time.sleep(10)  # 睡一会儿
                            # 不要更新 current_version！
                            # 下次循环回来，remote_version 依然 > current_version
                            # Client 会重新掷骰子，说不定下次就决定上线了
                            continue

                        # 3. 【新增】尝试报名 (Join Round)
                        self.logger.info(f"Detected new version {remote_version}. Attempting to join...")
                        
                        # 向 Edge 发送报名请求
                        join_resp = self.comm.post_data(f"{self.edge_url}/join_round", {"id": self.id})
                        
                        if join_resp and join_resp.status_code == 200:
                            # 报名成功！
                            self.logger.info("Join accepted! Downloading model...")
                            latest_weights = response['weights']
                            current_version = remote_version
                            # 跳出内层循环，进入 PHASE 2 训练
                            break
                        else:
                            # 报名失败（可能是招募窗口已关闭，或者 Edge 处于 TRAINING 状态）
                            # 这种情况下不更新 current_version，稍微等待后重试，
                            # 或者等待 Edge 下一次开启招募（版本号更新）
                            self.logger.warning("Join rejected (Window closed or Error). Waiting...")
                            time.sleep(2) 
                    else:
                        # 版本没变，继续等待
                        time.sleep(1)
                        
                except Exception as e:
                    self.logger.error(f"Connection error: {e}")
                    time.sleep(3)

            # ================= PHASE 2: 本地训练 =================
            self.logger.info(f"Starting training for round (v{current_version})...")
            
            # 加载权重
            self.model_mgr.load_weights(latest_weights)
            
            # 执行训练
            local_weights, train_loss, sample_count = self.model_mgr.train(self.train_loader)
            
            # ================= PHASE 3: 上传结果 =================
            upload_data = {
                "id": self.id,
                "weights": local_weights,
                "samples": sample_count,
                "version": current_version # 告诉 Edge 我是基于哪个版本训练的
            }
            
            self.logger.info("Uploading local update...")
            self.comm.post_data(f"{self.edge_url}/upload_client_model", upload_data)
            self.logger.info("Upload successful. Entering wait state.")