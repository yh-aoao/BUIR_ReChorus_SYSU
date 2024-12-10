import networkx as nx
import matplotlib.pyplot as plt

# 导入必要的模块和自定义的类与函数
from Models.BUIR_ID import BUIR_ID  # 导入BUIR_ID模型
from Models.BUIR_NB import BUIR_NB  # 导入BUIR_NB模型

from Utils.data_loaders import ImplicitFeedback  # 数据加载器
from Utils.data_utils import load_dataset, build_adjmat  # 工具函数：加载数据集和构建邻接矩阵
from Utils.evaluation import evaluate, print_eval_results  # 评估函数和打印结果函数

import torch  # PyTorch库
from torch.utils.data import DataLoader  # PyTorch的数据加载器

import numpy as np  # 数值计算库
import argparse  # 命令行参数解析
import os, time  # 操作系统接口和时间模块


def visualize_graph(edge_index, user_count, item_count, embeddings=None, user_item_pairs=None):
    G = nx.Graph()

    # Add nodes and edges to the graph
    for src, dst in zip(edge_index[0], edge_index[1]):
        G.add_edge(src.item(), dst.item())

    # Add node attributes
    for node in G.nodes():
        if node < user_count:
            G.nodes[node]['type'] = 'user'
        else:
            G.nodes[node]['type'] = 'item'

    # Create position layout for nodes
    pos = nx.spring_layout(G, k=0.5, seed=42)  # Increase k to spread out nodes

    # Customize node appearance based on type
    user_nodes = [node for node, attr in G.nodes(data=True) if attr['type'] == 'user']
    item_nodes = [node for node, attr in G.nodes(data=True) if attr['type'] == 'item']

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, nodelist=user_nodes, node_color='skyblue', label='Users', node_size=100)
    nx.draw_networkx_nodes(G, pos, nodelist=item_nodes, node_color='orange', label='Items', node_size=100)

    # Draw edges
    nx.draw_networkx_edges(G, pos, width=0.5, alpha=0.5)

    # Highlight specific user-item pairs
    if user_item_pairs is not None:
        highlighted_edges = [(user, item + user_count) for user, item in user_item_pairs]
        nx.draw_networkx_edges(G, pos, edgelist=highlighted_edges, color='red', width=2, alpha=0.8)

    # Draw labels for a subset of nodes to avoid clutter
    labels = {}
    for node in G.nodes():
        if node < user_count and node % 5 == 0:  # Display every 5th user node
            labels[node] = f'User {node}'
        elif node >= user_count and (node - user_count) % 5 == 0:  # Display every 5th item node
            labels[node] = f'Item {node - user_count}'

    nx.draw_networkx_labels(G, pos, labels, font_size=8, font_family='sans-serif')

    # Remove axis
    plt.axis('off')

    # Set figure size
    plt.figure(figsize=(16, 12))

    # Save the figure
    output_path = 'graph.png'
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"Graph saved to {output_path}")

    # Show the plot
    plt.show()


def run(args):
    """
    主运行函数，用于根据命令行参数配置来训练模型。
    """

    path = os.path.join(args.path, args.dataset)  # 构建数据集路径
    filename = 'users.dat'  # 数据文件名

    # 加载数据集，分割为训练、验证和测试集
    user_count, item_count, train_mat, valid_mat, test_mat = load_dataset(
        path, filename, train_ratio=args.train_ratio, random_seed=args.random_seed)

    # 创建一个数据集实例，并用DataLoader创建一个迭代器
    train_dataset = ImplicitFeedback(user_count, item_count, train_mat)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # 根据命令行参数选择模型类型并初始化
    if args.model == 'buir-id':
        model = BUIR_ID(user_count, item_count, args.latent_size, args.momentum)
    elif args.model == 'buir-nb':
        norm_adjmat = build_adjmat(user_count, item_count, train_mat, selfloop_flag=False)
        model = BUIR_NB(user_count, item_count, args.latent_size, norm_adjmat, args.momentum,
                        n_layers=args.n_layers, drop_flag=args.drop_flag)

    # 如果有GPU可用，则将模型移动到GPU上
    device = torch.device(f'cuda:{args.gpu}' if args.gpu >= 0 and torch.cuda.is_available() else 'cpu')
    model.to(device)

    # 定义优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_score = -np.inf  # 初始化最佳分数
    early_stop_cnt = 0  # 早停计数器

    # 记录训练和验证损失
    train_losses = []
    valid_scores = []

    # 开始训练循环
    for epoch in range(args.max_epochs):
        tic1 = time.time()  # 记录开始时间

        model.train()
        train_loss = []  # 存储每个batch的损失
        for batch in train_loader:
            # 将批次数据移动到设备上
            batch = {key: value.to(device) for key, value in batch.items()}

            # 前向传播
            output = model(batch)
            batch_loss = model.get_loss(output)
            train_loss.append(batch_loss.item())

            # 反向传播和优化
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            # 更新目标网络（如果有的话）
            model._update_target()

        # 计算平均训练损失
        avg_train_loss = sum(train_loss) / len(train_loss)
        train_losses.append(avg_train_loss)
        toc1 = time.time()  # 记录结束时间

        # 每10个epoch进行一次评估
        if epoch % 10 == 0 or epoch == args.max_epochs - 1:
            is_improved = False

            # 评估模型
            model.eval()
            with torch.no_grad():
                tic2 = time.time()
                eval_results = evaluate(model, train_loader, train_mat, valid_mat, test_mat)
                toc2 = time.time()

            # 检查是否改进
            if eval_results['valid']['P50'] > best_score:
                is_improved = True
                best_score = eval_results['valid']['P50']
                valid_result = eval_results['valid']
                test_result = eval_results['test']

                # 打印当前epoch的信息
                print('Epoch [{}/{}]'.format(epoch, args.max_epochs))
                print('Training Loss: {:.4f}, Elapsed Time for Training: {:.2f}s, for Testing: {:.2f}s\n'.format(
                    avg_train_loss, toc1 - tic1, toc2 - tic2))
                print_eval_results(eval_results)

            else:
                early_stop_cnt += 1
                if early_stop_cnt == args.early_stop:
                    print("EARLY_STOP：epoch = ", epoch)
                    break

            valid_scores.append(best_score)

    # 打印最终性能
    print('===== [FINAL PERFORMANCE] =====\n')
    print_eval_results({'valid': valid_result, 'test': test_result})

    # 绘制训练和验证曲线
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(valid_scores, label='Validation Score')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Validation Score Over Epochs')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.show()

    # Visualize the graph after training
    edge_index = model.online_encoder.edge_index.cpu().numpy()
    visualize_graph(edge_index, user_count, item_count)


def str2bool(v):
    """
    将字符串转换为布尔值。
    """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Invalid boolean value')


if __name__ == '__main__':
    # 设置命令行参数解析
    parser = argparse.ArgumentParser()

    # 模型相关的参数
    parser.add_argument('--model', type=str, default='buir-nb', help='选择模型类型：buir-id | buir-nb')
    parser.add_argument('--latent_size', type=int, default=250, help='隐层大小')
    parser.add_argument('--n_layers', type=int, default=3, help='图卷积层数')
    parser.add_argument('--drop_flag', type=str2bool, default=False, help='是否使用dropout')

    # 训练相关的参数
    parser.add_argument('--batch_size', type=int, default=512, help='批量大小')  # 调整批量大小
    parser.add_argument('--lr', type=float, default=0.0005, help='学习率')  # 调整学习率
    parser.add_argument('--weight_decay', type=float, default=0.00001, help='权重衰减')  # 调整权重衰减
    parser.add_argument('--momentum', type=float, default=0.99, help='动量')  # 调整动量
    parser.add_argument('--max_epochs', type=int, default=500, help='最大训练轮数')
    parser.add_argument('--early_stop', type=int, default=15, help='早停轮数')

    # 数据集相关的参数
    parser.add_argument('--path', type=str, default='./Data/', help='数据集路径')
    parser.add_argument('--dataset', type=str, default='toy-dataset', help='数据集名称')
    parser.add_argument('--train_ratio', type=float, default=0.7, help='训练集比例')  # 调整训练集比例
    parser.add_argument('--random_seed', type=int, default=42, help='随机种子')  # 更改随机种子

    # GPU设置
    parser.add_argument('--gpu', type=int, default=-1, help='使用的GPU编号 (-1 表示不使用GPU)')

    # 解析命令行参数
    args = parser.parse_args()

    # 设置GPU环境
    if args.gpu >= 0:
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    # 设置随机种子以保证实验可复现性
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    if args.gpu >= 0:
        torch.cuda.manual_seed_all(args.random_seed)

    # 调用主运行函数
    run(args)


