# 导入必要的模块和自定义的类与函数

from Models.BUIR_ID import BUIR_ID  # 导入BUIR_ID模型
from Models.BUIR_NB import BUIR_NB   # 导入BUIR_NB模型

from Utils.data_loaders import ImplicitFeedback  # 数据加载器
from Utils.data_utils import load_dataset, build_adjmat  # 工具函数：加载数据集和构建邻接矩阵
from Utils.evaluation import evaluate, print_eval_results  # 评估函数和打印结果函数

import torch  # PyTorch库
from torch.utils.data import DataLoader  # PyTorch的数据加载器

import numpy as np  # 数值计算库
import argparse  # 命令行参数解析
import os, time  # 操作系统接口和时间模块

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
    # model = model.cuda()

    # 定义优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_score = -np.inf  # 初始化最佳分数
    early_stop_cnt = 0  # 早停计数器

    # 开始训练循环
    for epoch in range(args.max_epochs):
        tic1 = time.time()  # 记录开始时间

        train_loss = []  # 存储每个batch的损失
        for batch in train_loader:
            # 将批次数据移动到GPU（如果有的话）
            # batch = {key: value.cuda() for key, value in batch.items()}
            batch = {key: value for key, value in batch.items()}  # 如果没有GPU则不移动

            # 前向传播
            model.train()
            output = model(batch)
            batch_loss = model.get_loss(output)
            train_loss.append(batch_loss)

            # 反向传播和优化
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            # 更新目标网络（如果有的话）
            model._update_target()

        # 计算平均训练损失
        train_loss = torch.mean(torch.stack(train_loss)).data.cpu().numpy()
        toc1 = time.time()  # 记录结束时间

        # 每10个epoch进行一次评估
        if epoch % 10 == 0:
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
                    train_loss, toc1-tic1, toc2-tic2))
                print_eval_results(eval_results)

            else:
                early_stop_cnt += 1
                if early_stop_cnt == args.early_stop:
                    print("EARLY_STOP：epoch = ", epoch)
                    break

    # 打印最终性能
    print('===== [FINAL PERFORMANCE] =====\n')
    print_eval_results({'valid': valid_result, 'test': test_result})

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
    parser.add_argument('--model', type=str, default='buir-id', help='选择模型类型：buir-id | buir-nb')
    parser.add_argument('--latent_size', type=int, default=250, help='隐层大小')
    parser.add_argument('--n_layers', type=int, default=3, help='图卷积层数')
    parser.add_argument('--drop_flag', type=str2bool, default=False, help='是否使用dropout')

    # 训练相关的参数
    parser.add_argument('--batch_size', type=int, default=1024, help='批量大小')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='权重衰减')
    parser.add_argument('--momentum', type=float, default=0.995, help='动量')
    parser.add_argument('--max_epochs', type=int, default=1000, help='最大训练轮数')
    parser.add_argument('--early_stop', type=int, default=10, help='早停轮数')

    # 数据集相关的参数
    parser.add_argument('--path', type=str, default='./Data/', help='数据集路径')
    parser.add_argument('--dataset', type=str, default='toy-dataset', help='数据集名称')
    parser.add_argument('--train_ratio', type=float, default=0.5, help='训练集比例')
    parser.add_argument('--random_seed', type=int, default=0, help='随机种子')

    # GPU设置
    parser.add_argument('--gpu', type=int, default=0, help='使用的GPU编号')

    # 解析命令行参数
    args = parser.parse_args()

    # 设置GPU环境
    # os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    # 设置随机种子以保证实验可复现性
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    # 调用主运行函数
    run(args)