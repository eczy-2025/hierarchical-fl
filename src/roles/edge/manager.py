import time
import threading
from src.modules.communication import FLCommunication
from src.roles.edge.model_mgr import EdgeModelManager

class EdgeManager:
    def __init__(self, config, edge_id, port, cloud_addr, cloud_port, logger, num_clients=2):
        self.config = config
        self.id = edge_id
        self.port = port
        self.cloud_addr = cloud_addr
        self.cloud_port = cloud_port
        self.logger = logger
        
        self.comm = FLCommunication(port=port, logger=logger)
        self.model_mgr = EdgeModelManager(config, logger, expected_clients=num_clients)
        
        self.internal_rounds_target = config['training'].get('edge_internal_rounds', 1)
        # 【新增】读取招募等待时间和最少参与人数
        self.recruitment_time = config['training'].get('recruitment_time', 5)
        self.min_clients = config['training'].get('min_clients', 1)
        
        self.current_internal_round = 0
        
        # 【新增】招募状态管理
        self.status = "IDLE" # IDLE, RECRUITING, TRAINING
        self.participants = set() # 本轮报名的客户端ID
        self.status_lock = threading.Lock()
        
        self._register_routes()

    def _register_routes(self):
        # 【新增】客户端报名接口
        def handle_join():
            from flask import request
            try:
                data = self.comm.deserialize_request(request)
                client_id = data.get('id')
                
                with self.status_lock:
                    if self.status == "RECRUITING":
                        self.participants.add(client_id)
                        self.logger.info(f"Client {client_id} joined the round.")
                        return "OK"
                    else:
                        # 如果不在招募期，拒绝加入
                        return "REJECT", 400
            except Exception as e:
                self.logger.error(f"Join error: {e}")
                return "ERR", 500

        def handle_upload():
            from flask import request
            try:
                data = self.comm.deserialize_request(request)
                client_id = data.get('id')
                
                # 【最终安全补丁】严格校验身份
                # 只有在当前轮次成功报名 (participants) 的客户端，才允许上传
                with self.status_lock:
                    if client_id not in self.participants:
                        self.logger.warning(f"Refused update from non-participant Client {client_id}")
                        return "REJECT", 403
                
                self.model_mgr.add_client_update(data)
                self.logger.info(f"Received update from Client {client_id}")
                return "ACK"
            except Exception as e:
                self.logger.error(f"Upload error: {e}")
                return "ERR", 500

        def handle_download():
            data = self.model_mgr.get_latest_model()
            if data is None:
                return "NOT_READY", 404
            return self.comm.pickle_response(data)

        self.comm.add_endpoint('/join_round', 'join', handle_join, methods=['POST'])
        self.comm.add_endpoint('/upload_client_model', 'upload', handle_upload, methods=['POST'])
        self.comm.add_endpoint('/get_latest_model', 'download', handle_download, methods=['GET'])

    def start_recruitment(self):
        """开启招募窗口，返回是否招募成功"""
        with self.status_lock:
            self.status = "RECRUITING"
            self.participants.clear()
	    # 【新增】确保新一轮招募开始时，没有任何残留的历史数据
            # 需要在 EdgeModelManager 中加一个 clear_buffer 方法，或者直接在这里访问
            self.model_mgr.received_updates = []
        
        self.logger.info(f"📢 Recruitment started. Waiting {self.recruitment_time}s for clients...")
        time.sleep(self.recruitment_time)
        
        with self.status_lock:
            self.status = "TRAINING"
            num_joined = len(self.participants)
            
            # 检查是否满足最小参与人数
            if num_joined < self.min_clients:
                self.logger.warning(f"Not enough clients ({num_joined} < {self.min_clients}). Retry needed.")
                # 设置一个不可能达到的阈值，防止意外聚合
                self.model_mgr.reset_threshold(9999)
                return False
            
            # 动态设置阈值
            self.model_mgr.reset_threshold(num_joined)
            self.logger.info(f"🏁 Recruitment closed. Participants: {num_joined}. Threshold set to {num_joined}.")
            return True

    def start(self):
        self.comm.start_server_thread()
        self.logger.info(f"Edge {self.id} started. Strategy: Recruitment window {self.recruitment_time}s.")
        
        cloud_get_url = f"http://{self.cloud_addr}:{self.cloud_port}/get_global_model"
        cloud_upload_url = f"http://{self.cloud_addr}:{self.cloud_port}/upload_edge_model"
        
        while True:
            # A. [Cloud Sync]
            # 只有在内部轮次归零（等待新一轮开始）时，才尝试从 Cloud 拉取
            if self.current_internal_round == 0:
                try:
                    response = self.comm.get_data(cloud_get_url)
                    if response:
                        updated = self.model_mgr.load_global_weights(response)
                        if updated:
                            self.logger.info(f"Pulled new global model (Round {response['global_round']}).")
                            
                            # 新模型已就绪，开始招募
                            # 如果招募失败（人数不足），循环重试，直到满足条件才进入训练阶段
                            while not self.start_recruitment():
                                self.logger.info("Retrying recruitment in 2s...")
                                time.sleep(2)
                                
                except Exception:
                    pass

            # B. [Aggregation]
            if self.model_mgr.check_aggregation_condition():
                self.logger.info("All participants reported. Aggregating...")
                
                result = self.model_mgr.aggregate()
                if result:
                    self.current_internal_round += 1
                    
                    if self.current_internal_round < self.internal_rounds_target:
                        # 本地迭代：更新本地模型
                        self.model_mgr.load_global_weights({'weights': result['weights']})
                        self.logger.info(f"Internal round finished ({self.current_internal_round}/{self.internal_rounds_target}). Version updated.")
                        
                        # 【关键】每一轮内部迭代开始前，都要重新招募！
                        # 因为上一轮在线的 Client 这一轮可能掉线
                        while not self.start_recruitment():
                             self.logger.info("Retrying recruitment for internal round...")
                             time.sleep(2)

                    else:
                        # 达到目标：上传云端
                        upload_data = {
                            "id": self.id,
                            "weights": result['weights'],
                            "samples": result['samples']
                        }
                        self.comm.post_data(cloud_upload_url, upload_data)
                        self.logger.info(f"Target reached. Uploaded to Cloud.")
                        
                        # 重置
                        self.current_internal_round = 0
                        with self.status_lock:
                            self.status = "IDLE"

            time.sleep(1)