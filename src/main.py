import argparse
import time
from src.utils import load_config, setup_seed
from src.modules.logger import setup_logger
from src.roles.cloud.manager import CloudManager
from src.roles.edge.manager import EdgeManager
from src.roles.client.manager import ClientManager

def parse_args():
    parser = argparse.ArgumentParser(description="Hierarchical FL Simulation")
    
    # 通用参数
    parser.add_argument('--role', type=str, required=True, choices=['cloud', 'edge', 'client'], help='Role to start')
    parser.add_argument('--config', type=str, default='configs/global_config.yaml', help='Path to config file')
    parser.add_argument('--id', type=int, default=0, help='ID of the node')
    
    # 网络参数
    parser.add_argument('--port', type=int, help='Port to listen on (for Cloud/Edge)')
    parser.add_argument('--cloud_addr', type=str, help='Address of Cloud Server')
    parser.add_argument('--cloud_port', type=int, help='Port of Cloud Server')
    parser.add_argument('--edge_addr', type=str, help='Address of Edge Server (for Client)')
    parser.add_argument('--edge_port', type=int, help='Port of Edge Server (for Client)')
    
    # 【新增】Edge 参数：该节点下辖的 Client 数量
    parser.add_argument('--num_clients', type=int, default=1, help='Number of clients managed by this edge node')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. 加载配置与初始化
    config = load_config(args.config)
    setup_seed(config['system']['random_seed'])
    
    # 2. 设置日志
    log_file = f"logs/{args.role.upper()}_{args.id}.log"
    logger = setup_logger(f"{args.role.upper()}_{args.id}", log_file)
    
    logger.info(f"🚀 Starting {args.role.upper()} node (ID: {args.id})...")
    
    # 3. 根据角色启动对应的 Manager
    if args.role == 'cloud':
        if not args.port: args.port = config['network']['cloud_port']
        manager = CloudManager(config, args.port, logger)
        manager.start()
        
    elif args.role == 'edge':
        if not args.port: raise ValueError("Edge requires --port")
        if not args.cloud_addr: raise ValueError("Edge requires --cloud_addr")
        if not args.cloud_port: args.cloud_port = config['network']['cloud_port']
        
        # 【修改】将 num_clients 传入 Manager
        manager = EdgeManager(
            config, args.id, args.port, 
            args.cloud_addr, args.cloud_port, logger,
            num_clients=args.num_clients
        )
        manager.start()
        
    elif args.role == 'client':
        if not args.edge_addr: raise ValueError("Client requires --edge_addr")
        if not args.edge_port: raise ValueError("Client requires --edge_port")
        
        manager = ClientManager(
            config, args.id, 
            args.edge_addr, args.edge_port, logger
        )
        manager.start()

if __name__ == '__main__':
    main()