# Dockerfile

# 1. 使用轻量级的 Python 3.9 基础镜像
FROM python:3.9-slim

# 2. 设置工作目录
WORKDIR /app

# ==========================================
# 【修正点】针对 Debian 12 (Bookworm) 换源
# 配置文件变成了 /etc/apt/sources.list.d/debian.sources
# ==========================================
RUN sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list.d/debian.sources

# 3. 安装编译依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 4. 复制依赖文件
COPY requirements.txt .

# 5. 使用阿里云 PyPI 镜像源安装 Python 包
RUN pip install --default-timeout=10000 -i https://pypi.tuna.tsinghua.edu.cn/simple/ --no-cache-dir -r requirements.txt

# 6. 复制源代码和配置
COPY src/ ./src/
COPY configs/ ./configs/

# 7. 创建必要的空文件夹
RUN mkdir -p /app/data /app/logs /app/output

# 8. 设置环境变量
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# 【新增】将根目录加入 Python 搜索路径，彻底解决 ModuleNotFoundError
ENV PYTHONPATH=/app

# 9. 入口点
# 【删除或注释掉这一行】因为 docker-compose.yml 里已经写了完整的启动命令
# ENTRYPOINT ["python", "src/main.py"]
