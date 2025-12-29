# 层级联邦学习 (HFL) 仿真系统设计说明书

## 1. 系统概述

本系统是一个基于 Docker 容器化技术的层级联邦学习（Hierarchical Federated Learning, HFL）仿真平台。系统严格遵循 **Cloud-Edge-Client（云-边-端）** 三层物理架构设计。

**核心特性：**

* **分层架构**：支持 Cloud-Edge-Client 三层拓扑。
* **通信优化**：支持 Edge 端本地多轮聚合 (Internal Rounds)，减少广域网通信。
* **动态拓扑 (New)**：支持 **Edge 端动态招募** 与 **Client 端随机掉线/重连**，模拟真实不稳定的网络环境。
* **鲁棒性 (New)**：具备自愈能力，能够处理节点掉线、参与度不足重试以及僵尸节点过滤。

## 2. 系统总体架构与流程

系统采用 **“输入-处理-输出”** 的闭环设计模式，并引入了 **“招募-训练-聚合”** 的动态调度流程。

### 2.1 输入层：全局配置 (Global Configuration)

系统通过统一的配置文件初始化仿真环境。

* **配置文件**: `configs/global_config.yaml`
* **核心配置**:
* **数据分布**: Non-IID 分布控制（Dirichlet ）。
* **动态参数**: `online_rate` (在线率), `recruitment_time` (招募窗口), `min_clients` (最少参与人数)。



### 2.2 核心处理层：层级架构 (Hierarchical Architecture)

#### A. 云服务器层 (Cloud Server Layer)

* **对应容器**: `cloud_server`
* **核心功能**:
1. **全局模型管理**: 维护全局模型与版本 (Global Round)。
2. **轮次控制**: 仅在收到 Edge 的聚合结果后增加全局轮次，驱动系统向前演进。
3. **全局评估**: 聚合后在测试集上评估模型精度。



#### B. 边缘服务器层 (Edge Server Layer)

* **对应容器**: `edge_server_x`
* **核心功能**:
1. **动态招募 (Recruitment Phase)**:
* 开启新一轮时，启动 `recruitment_time` 秒的倒计时窗口。
* 接受 Client 的 `/join_round` 报名请求。
* **动态阈值**: 窗口结束后，根据实际报名人数动态调整聚合阈值 (Threshold)。


2. **自愈重试 (Retry Loop)**:
* 若招募到的 Client 数量不足 `min_clients`，自动进入休眠并开启下一轮招募，防止死锁。


3. **安全过滤 (Zombie Filter)**:
* 拒绝非本轮报名名单内的 Client 提交更新，防止过时数据污染模型。


4. **本地多轮迭代**: 执行 `edge_internal_rounds` 次本地聚合后，才向 Cloud 上传。



#### C. 客户端层 (Client Layer)

* **对应容器**: `client_x`
* **核心功能**:
1. **随机在线 (Simulated Dropout)**:
* 检测到新版本后，根据 `online_rate` 概率决定本轮是否“在线”。
* 若决定“掉线”，则休眠并跳过本轮，等待下一次机会。


2. **主动报名 (Active Joining)**:
* 决定在线后，向 Edge 发送 `/join_round` 请求。
* 仅在报名成功后下载模型并开始训练。


3. **数据隔离**: 拥有互不重叠的本地私有数据集。



### 2.3 输出层：结果可视化 (Output)

* **输出**: 标准输出日志 (Logs) 与 `output/` 目录下的实验图表。

---

## 3. 代码目录结构设计

```text
hierarchical-fl/
├── configs/
│   └── global_config.yaml     # 【输入】全局配置文件 (含动态招募参数)
├── data/                      # 数据集挂载点
├── logs/                      # 【输出】训练日志统一存放点
├── output/                    # 【输出】实验结果图表
├── src/
│   ├── modules/               # 通用基础模块 (Communication, DataSlicer, Model)
│   ├── roles/                 # 角色逻辑实现
│   │   ├── cloud/             # Cloud: 全局聚合
│   │   ├── edge/              # Edge: 招募窗口、动态阈值、内部聚合
│   │   │   ├── manager.py     # 业务逻辑 (招募状态机)
│   │   │   └── model_mgr.py   # 模型管理 (动态重置阈值)
│   │   └── client/            # Client: 随机掉线、主动报名
│   │       └── manager.py     # 业务逻辑 (状态轮询)
│   ├── main.py                # 系统启动入口
│   └── utils.py
├── Dockerfile                 # 容器构建文件
└── docker-compose.yml         # 物理拓扑编排

```

---

## 4. 网络拓扑与部署

系统利用 Docker Network 实现层级隔离。在 `docker-compose.yml` 中，Edge 节点通过参数定义其辖区，但聚合逻辑已升级为动态适应实际在线人数。

---

## 5. 快速开始指南

### 前置要求

确保本机已安装 Docker 和 Docker Compose。

### 启动步骤

1. **清理旧容器** (防止命名冲突):
```bash
sudo docker rm -f hfl_cloud hfl_edge_1 hfl_edge_2 hfl_client_1_1 hfl_client_1_2 hfl_client_2_1 hfl_client_2_2

```


2. **构建并启动**:
```bash
sudo docker compose up --build

```


3. **查看实时日志**:
```bash
sudo docker compose logs -f

```



### 预期日志流 (Feature Validation)

您将在日志中观察到以下特征，证明系统的动态特性正在运行：

1. **Edge 开启招募**:
```text
[EDGE_1] 📢 Recruitment started. Waiting 15s for clients...

```


2. **Client 随机决策**:
```text
[CLIENT_101] Detected new version. Attempting to join...
[CLIENT_102] Feature: Client decided to be OFFLINE... Sleeping.  <-- 模拟掉线

```


3. **动态阈值设定**:
```text
[EDGE_1] Client 101 joined the round.
[EDGE_1] 🏁 Recruitment closed. Participants: 1. Threshold set to 1. <-- 自动适应在线人数

```


4. **异常处理 (如无人报名)**:
```text
[EDGE_1] Not enough clients (0 < 1). Retry needed.
[EDGE_1] Retrying recruitment in 2s... <-- 自愈重试

```



---

## 6. 配置说明 (global_config.yaml)

主要可调参数说明：

| 参数模块 | 参数名 | 说明 |
| --- | --- | --- |
| **dataset** | `partition` | 设置为 `"noniid"` 开启非独立同分布模拟。 |
|  | `beta` | 控制数据异构程度（越小越不平衡）。 |
| **training** | `global_rounds` | 云端总聚合次数。 |
|  | `edge_internal_rounds` | Edge 在上传 Cloud 前，本地聚合的次数。 |
|  | **`online_rate`** | **【新】** 客户端在线概率 (0.0 ~ 1.0)。 |
|  | **`recruitment_time`** | **【新】** Edge 等待报名的窗口时间 (秒)，建议设为 10-15s 以适应 Docker 网络延迟。 |
|  | **`min_clients`** | **【新】** 每一轮最少需要多少个客户端报名才开启训练，否则重试。 |

