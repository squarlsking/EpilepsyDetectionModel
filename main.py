"""
癫痫发作预测系统 - 主入口
"""

import argparse
import sys
import os

def main():
    parser = argparse.ArgumentParser(
        description="癫痫发作预测系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 训练模型
  python main.py train --data data/rr_dataset.csv --output models/model.pkl
  
  # 启动服务器
  python main.py serve --port 8000
  
  # 可视化数据
  python main.py visualize --data data/sample_dataset.csv --output output/
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # 训练命令
    train_parser = subparsers.add_parser('train', help='训练HRV分类模型')
    train_parser.add_argument('--data', required=True, help='训练数据CSV文件路径')
    train_parser.add_argument('--output', default='models/model.pkl', help='输出模型路径')
    
    # 服务器命令
    serve_parser = subparsers.add_parser('serve', help='启动Flask API服务器')
    serve_parser.add_argument('--port', type=int, default=8000, help='服务器端口')
    serve_parser.add_argument('--host', default='0.0.0.0', help='服务器主机')
    
    # 可视化命令
    viz_parser = subparsers.add_parser('visualize', help='可视化癫痫事件数据')
    viz_parser.add_argument('--data', required=True, help='数据CSV文件路径')
    viz_parser.add_argument('--output', default='output/', help='输出目录')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # 执行对应的命令
    if args.command == 'train':
        print(f"🚀 开始训练模型...")
        print(f"   数据: {args.data}")
        print(f"   输出: {args.output}")
        os.system(f'python scripts/train_hrv_model.py "{args.data}" "{args.output}"')
        
    elif args.command == 'serve':
        print(f"🌐 启动服务器...")
        print(f"   地址: http://{args.host}:{args.port}")
        os.environ['PORT'] = str(args.port)
        os.system(f'python scripts/hrv_server.py')
        
    elif args.command == 'visualize':
        print(f"📊 开始可视化...")
        print(f"   数据: {args.data}")
        print(f"   输出: {args.output}")
        os.system(f'python visualization/epilepsy_event_plotter.py "{args.data}" "{args.output}"')

if __name__ == "__main__":
    main()
