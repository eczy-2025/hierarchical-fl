import logging
import os
import sys

def setup_logger(name, log_file, level=logging.INFO):
    """
    配置并返回一个 logger 对象
    :param name: Logger 的名字 (e.g., 'CLOUD', 'CLIENT_101')
    :param log_file: 日志文件保存路径
    :param level: 日志级别
    """
    # 1. 创建 logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 避免重复添加 handler (Flask 重载时容易出现)
    if logger.handlers:
        return logger

    # 2. 定义格式: [时间] [名字] 信息
    formatter = logging.Formatter(
        '[%(asctime)s] [%(name)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 3. Handler 1: 输出到控制台
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # 4. Handler 2: 输出到文件
    # 确保目录存在
    log_dir = os.path.dirname(log_file)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
