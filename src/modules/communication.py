import pickle
import requests
import threading
from flask import Flask, request, Response
import logging

# 屏蔽 Flask 默认的启动广告，让日志更干净
import click
def secho(text, file=None, nl=None, err=None, color=None, **styles):
    pass
def echo(text, file=None, nl=None, err=None, color=None, **styles):
    pass
click.echo = echo
click.secho = secho

class FLCommunication:
    def __init__(self, host='0.0.0.0', port=5000, logger=None):
        self.app = Flask(__name__)
        self.host = host
        self.port = port
        self.logger = logger
        
        # 禁用 Flask 默认日志
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.ERROR)

    def add_endpoint(self, endpoint, endpoint_name, handler_func, methods=['POST', 'GET']):
        """注册 API 接口"""
        self.app.add_url_rule(endpoint, endpoint_name, handler_func, methods=methods)

    def start_server(self):
        """启动 HTTP 服务器 (阻塞式)"""
        if self.logger:
            self.logger.info(f"Server starting at http://{self.host}:{self.port}")
        # threaded=True 允许并发请求
        self.app.run(host=self.host, port=self.port, threaded=True)

    def start_server_thread(self):
        """在子线程中启动服务器 (非阻塞)"""
        t = threading.Thread(target=self.start_server, daemon=True)
        t.start()

    # ================= 通信核心方法 (改为实例方法以支持 logging) =================

    def post_data(self, target_url, data_dict):
        """
        发送数据 (POST)
        :return: requests.Response 对象 (成功时) 或 None (失败时)
        """
        try:
            binary_data = pickle.dumps(data_dict)
            response = requests.post(
                target_url, 
                data=binary_data, 
                headers={'Content-Type': 'application/octet-stream'},
                timeout=30 # 设置超时防止死锁
            )
            return response
        except Exception as e:
            # 捕获连接错误，方便调试 Client 加入失败的原因
            if self.logger:
                self.logger.error(f"Post error to {target_url}: {e}")
            return None

    def get_data(self, target_url):
        """拉取数据 (GET)"""
        try:
            response = requests.get(target_url, timeout=30)
            if response.status_code == 200:
                return pickle.loads(response.content)
            else:
                return None
        except Exception as e:
            if self.logger:
                self.logger.error(f"Get error from {target_url}: {e}")
            return None

    def pickle_response(self, data):
        """将对象打包成 Flask 响应"""
        return Response(pickle.dumps(data), mimetype='application/octet-stream')

    def deserialize_request(self, flask_request):
        """从 Flask 请求中解析对象"""
        return pickle.loads(flask_request.data)