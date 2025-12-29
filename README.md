# 层级联邦学习 (HFL) 仿真系统说明书

## 1. 系统概述

本系统是一个基于 Docker 容器化技术的层级联邦学习（Hierarchical Federated Learning, HFL）仿真平台。系统严格遵循 **Cloud-Edge-Client（云-边-端）** 三层物理架构设计，支持 **Edge端本地多轮聚合** 与 **Client端版本控制同步** 机制，旨在模拟真实网络环境下的低带宽通信与分布式训练流程。

## 2. 系统总体架构与流程

系统采用 **“输入-处理-输出”** 的闭环设计模式，并引入了严格的 **版本同步 (Version Sync)** 机制以确保训练时序的正确性。

### 2.1 输入层：全局配置 (Global Configuration)

系统通过统一的配置文件初始化仿真环境。

* **配置文件**: `configs/global_config.yaml`
* **数据分布**: 支持 Non-IID 分布（Dirichlet  参数控制）。
* **训练策略**:
* `edge_internal_rounds`: **【新特性】** Edge 在向 Cloud 上传前，在本地执行聚合的轮数。
* `local_epochs`: Client 本地训练轮数。


* **模型参数**: 定义网络结构、学习率 (LR)、批量大小 (Batch Size)。



### 2.2 核心处理层：层级架构 (Hierarchical Architecture)

#### A. 云服务器层 (Cloud Server Layer)

* **对应容器**: `cloud_server`
* **核心功能**:
1. **全局模型管理**: 生成初始模型，执行全局聚合 (Global Aggregation)。
2. **轮次控制 (Round Sync)**: 维护 `Global Round` ID。只有当聚合发生时，Global Round 才会增加，该 ID 用于同步下游 Edge 节点。
3. **全局评估**: 每次聚合后，在测试集上评估全局模型精度 (Accuracy)。



#### B. 边缘服务器层 (Edge Server Layer)

* **对应容器**: `edge_server_x`
* **核心功能**:
1. **拓扑感知聚合 (Topology-aware Aggregation)**:
* 启动时通过 `--num_clients` 参数感知下辖客户端数量。
* **严格同步**: 必须收到**所有**下辖 Client 的更新包后，才触发聚合，杜绝部分节点抢跑。


2. **本地多轮迭代 (Edge Internal Training)**:
* 执行 `edge_internal_rounds` 次本地聚合。
* 在此期间，Edge 仅更新本地模型版本并通知 Client 继续训练，**不**向 Cloud 上传。


3. **云端同步**: 主动轮询 Cloud，仅在发现 `Global Round` 增加时才更新本地模型。



#### C. 客户端层 (Client Layer)

* **对应容器**: `client_x`
* **核心功能**:
1. **版本控制与等待 (Blocking Wait)**:
* 采用 **被动训练模式**。上传完成后立即进入阻塞状态。
* 轮询 Edge 接口，仅当检测到模型 **Version** 更新（由 Edge 聚合完成触发）时，才下载新模型并开始下一轮训练。


2. **数据管理 (Unique Partitioning)**:
* 基于 Edge ID 和 Client ID 的组合哈希算法，确保即使在不同 Edge 下的 Client 也能获取互不重叠的数据切片。





### 2.3 输出层：结果可视化 (Output)

* **输出**: 标准输出日志 (Logs) 与 `output/` 目录下的实验图表。

---

## 3. 代码目录结构设计

```text
hierarchical-fl/
├── configs/
│   └── global_config.yaml     # 【输入】全局配置文件
├── data/                      # 数据集挂载点
├── logs/                      # 【输出】训练日志统一存放点
├── output/                    # 【输出】实验结果图表
├── src/
│   ├── modules/               # 通用基础模块 (Communication, Logger, Model, DataLoader)
│   ├── roles/                 # 角色逻辑实现
│   │   ├── cloud/             # Cloud: 全局聚合与轮次管理
│   │   ├── edge/              # Edge: 内部多轮聚合与客户端阈值管理
│   │   └── client/            # Client: 版本等待与本地训练
│   ├── main.py                # 系统启动入口
│   └── utils.py
├── Dockerfile                 # 容器构建文件
└── docker-compose.yml         # 物理拓扑编排

```

---

## 4. 网络拓扑与部署

系统利用 Docker Network 实现层级隔离，并通过 `docker-compose.yml` 定义物理拓扑。

* **拓扑配置关键点**:
在 `docker-compose.yml` 中，每个 Edge 节点启动时必须通过 `--num_clients` 参数指定其管理的 Client 数量：
```yaml
edge_server_1:
  command: >
    python3 -m src.main ... --num_clients 2  # Edge 1 关联 2 个 Client

```


这确保了聚合阈值是动态适应拓扑结构的。

---

## 5. 快速开始指南

### 前置要求

确保本机已安装 Docker 和 Docker Compose。

### 启动步骤

1. 进入项目根目录。
2. 构建并启动集群：
```bash
docker compose up --build

```


3. 查看实时日志流：
```bash
docker compose logs -f

```



### 预期日志流 (Feature Validation)

您将在日志中观察到以下特征，证明系统运行正常：

1. **Client 等待机制**:
```text
[CLIENT] Upload successful. Entering wait state.
[CLIENT] Waiting for new model from Edge...
[CLIENT] Received new global model (v2)  <-- 版本号变大，触发训练

```


2. **Edge 内部聚合**:
```text
[EDGE] All clients reported. Aggregating...
[EDGE] Internal round finished (1/3). Version updated.  <-- 本地聚合，不传云端
...
[EDGE] Internal round finished (3/3). Target reached. Uploaded to Cloud. <-- 达到设定轮数，上传云端

```


3. **云端同步**:
```text
[CLOUD] Received update from Edge... Global Aggregation finished.
[EDGE] Pulled new global model (Round X). <-- Edge 感知到云端轮次更新

```



---

## 6. 配置说明 (global_config.yaml)

主要可调参数说明：

* **`dataset.partition`**: 设置为 `"noniid"` 开启非独立同分布模拟。
* **`dataset.beta`**: 控制数据异构程度（越小越不平衡）。
* **`training.global_rounds`**: 云端总聚合次数。
* **`training.edge_internal_rounds`**: **【关键】** Edge 在向 Cloud 上传前，在本地通过与 Client 交互进行聚合的次数。



* **注意**: `edge_aggregation_step` 参数在当前版本中已被 `docker-compose.yml` 中的 `--num_clients` 启动参数覆盖，以实现更灵活的拓扑适配。



