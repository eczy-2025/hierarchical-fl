# 层级联邦学习 (HFL) 仿真系统设计说明书

## 1. 系统概述

本系统是一个基于 Docker 容器化技术的层级联邦学习（Hierarchical Federated Learning, HFL）仿真平台。系统严格遵循 **Cloud-Edge-Client（云-边-端）** 三层物理架构设计，旨在解决复杂异构网络中的“数据偏科”、“设备掉队”以及“广域网通信延迟”等核心痛点。

**核心算法特性：**

* **端边感知自适应同步 (Ada t&d-aware)**：边缘端具备“数据异构感知”能力，通过计算梯度散度动态为客户端分配个性化训练轮次（Epochs），限制偏科节点，鼓励优质节点。
* **边云混合异步聚合 (FedDoMA)**：云端采用面向动态性的多元异步聚合框架。摒弃传统的死等机制，引入基于“到达数量”与“目标更新距离”的**双通道动态触发阀门**，并利用包含“陈旧度(Staleness)”与“相似度(Similarity)”的**多元加权算法 (MultiAsyncAgg)** 消除异步误差。
* **动态拓扑与鲁棒性**：原生支持边缘动态招募与客户端随机掉线（Dropout）模拟，具备自愈与僵尸节点过滤能力。

---

## 2. 系统总体架构与核心算法流程

系统采用 **“预热摸底 -> 自适应分配 -> 局部同步 -> 全局异步”** 的复杂调度流程。

### 2.1 云服务器层 (Cloud Server Layer)

* **对应容器**: `cloud_server`
* **核心功能 (FedDoMA)**:
1. **动态异步触发阀门 (Dynamic Trigger)**：
* **数量触发**：更新队列模型数达到上限 $\bar{M}$（兜底防无限等待）。
* **距离触发**：预测的全局模型更新幅度（L2 范数距离）接近动态设定的目标距离（自适应收敛步幅）。


2. **多元异步聚合算法 (MultiAsyncAgg)**：
* **陈旧度惩罚 ($\alpha_j$)**：通过解析 Edge 附带的版本号计算延迟轮次，应用指数衰减函数 $(1+\tau_j)^{-a}$ 降低过时模型的权重。
* **相似度奖励 ($s_j$)**：计算边缘模型与全局模型更新方向的余弦相似度，赋予一致性高的模型更大权重。
* **防偏移修正**：在综合数据量、陈旧度与相似度算出权重 $\theta_j$ 后，引入修正项 $(1-\Phi)w^t$ 补偿模型参数尺度，确保稳定收敛。





### 2.2 边缘服务器层 (Edge Server Layer)

* **对应容器**: `edge_server_x`
* **核心功能 (Ada d-aware)**:
1. **预训练摸底 (Pre-training)**：在最初的几轮（如 Round 0-2），下发固定 Epoch，并在后台计算各客户端局部梯度与全局梯度的距离，估算**梯度散度 $\delta_i$**。
2. **自适应 Epoch 分配**：预热期结束后，利用启发式对数公式 $E_i = \lfloor \log(\frac{\phi}{\sqrt{\delta_i}} + 1) \rfloor$ 瞬间计算出专属迭代次数。数据越偏科（$\delta_i$ 大），分配的 $E_i$ 越小。
3. **动态招募与层级聚合**：开启倒计时招募窗口，收集存活节点的报名。执行完指定次数的 `edge_internal_rounds`（节省带宽）后，附带当前基于的 `version` 版本号发往云端。



### 2.3 客户端层 (Client Layer)

* **对应容器**: `client_x`
* **核心功能**:
1. **精准执行指令**：不再使用写死的本地轮次，而是通过解析二进制 `Pickle` 流，严格按照 Edge 为其量身定制的 `assigned_epochs` 进行本地训练。
2. **随机在线与自愈**：根据 `online_rate` 概率掷骰子决定本轮是否在线。
3. **异构数据持有**：基于 Dirichlet 分布切分的强 Non-IID 本地数据集。



---

## 3. 代码目录结构设计

```text
hierarchical-fl/
├── configs/
│   └── global_config.yaml     # 【输入】全局配置文件 (Non-IID分布, 掉线率等)
├── data/                      # 数据集挂载点
├── logs/                      # 训练日志统一存放点
├── src/
│   ├── modules/               # 通用基础模块 (Communication, DataSlicer, Model)
│   ├── roles/                 
│   │   ├── cloud/             
│   │   │   ├── manager.py     # 业务逻辑 (FedDoMA 双通道动态触发轮询)
│   │   │   └── model_mgr.py   # 模型大脑 (MultiAsyncAgg 算法, 余弦相似度, 陈旧度计算)
│   │   ├── edge/              
│   │   │   ├── manager.py     # 业务逻辑 (对数公式分配 Epochs, 携带 Version 上传)
│   │   │   └── model_mgr.py   # 模型大脑 (梯度散度 delta_i 计算与历史缓存)
│   │   └── client/            
│   │       ├── manager.py     # 业务逻辑 (解析 Pickle 获取定制 Epoch, 随机 Dropout)
│   │       └── model_mgr.py   # 执行层 (基于动态 Epochs 执行前向与反向传播)
│   └── main.py                # 系统启动入口
├── Dockerfile                 # 容器构建文件
└── docker-compose.yml         # 物理拓扑编排 (1 Cloud, 2 Edge, 4 Client)

```

---

## 4. 快速开始指南

### 启动步骤

1. **清理旧容器** (防止端口或命名冲突):

```bash
sudo docker rm -f hfl_cloud hfl_edge_1 hfl_edge_2 hfl_client_1_1 hfl_client_1_2 hfl_client_2_1 hfl_client_2_2

```

2. **构建并启动集群**:

```bash
sudo docker compose up --build

```

### 预期的高级算法日志 (Feature Validation)

本系统的核心创新将通过以下极具辨识度的日志展现：

**1. 端边层：数据感知与自适应分配 (Ada t&d-aware)**

```text
[EDGE_1] Client 101 joined. Assigned Epochs: 22 (Round: 4, delta_i: 0.483)  <-- 偏科限制
[EDGE_1] Client 102 joined. Assigned Epochs: 25 (Round: 4, delta_i: 0.203)  <-- 优质奖励
[CLIENT_102] Ada t&d-aware: Training for 25 epochs based on Edge instruction.

```

**2. 边云层：动态触发与多元加权 (FedDoMA)**

```text
[CLOUD_0] ⚡ FedDoMA Triggered! Reason: Max Queue Size Reached (2/2)
[CLOUD_0] Aggregating Edge models...
[CLOUD_0] [FedDoMA] Update 0 -> alpha: 1.0000, sim: 0.9839, theta: 0.3440  <-- 陈旧度与相似度打分
[CLOUD_0] [FedDoMA] Update 1 -> alpha: 1.0000, sim: 0.9859, theta: 0.6471
[CLOUD_0] === Global Round 4 Result ===
[CLOUD_0]     Accuracy : 82.34%

```

---

## 5. 核心超参数配置说明

除传统的通信轮次配置外，系统内嵌了以下与算法深度相关的超参数（可在源码或 YAML 中扩展调整）：

| 所属模块 | 参数 / 概念 | 说明 |
| --- | --- | --- |
| **Data Slicer** | `partition: "noniid"` | 开启狄利克雷分布，制造数据异质性（测试算法防御能力的先决条件）。 |
| **Client** | `online_rate` | 客户端在线概率 (如 0.8)，用于测试系统的防死锁与自愈鲁棒性。 |
| **Edge (Ada)** | $\phi$ (Phi) | 启发式对数公式的调节因子。$\phi$ 越大，分配的平均 Epoch 越高。 |
| **Cloud (FedDoMA)** | `max_queue_size` | 异步队列数量触发阈值 $\bar{M}$。 |
| **Cloud (FedDoMA)** | `stale_ratio` / `hetero_ratio` | 多元聚合时，陈旧度惩罚与相似度奖励所占的比重系数（默认 0.4 / 0.6）。 |