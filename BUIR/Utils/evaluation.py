import torch
import numpy as np
import copy 

from Utils.data_utils import to_np # 这是一个自定义模块，用于将张量转换为numpy数组


def evaluate(model, data_loader, train_mat, valid_mat, test_mat):
    """
    评估推荐系统模型在验证集和测试集上的表现。

    参数:
        model: 推荐系统模型，提供用户和物品的嵌入。
        data_loader: 数据加载器，通常用于迭代数据集，在本函数中未直接使用。
        train_mat: 训练集用户-物品交互矩阵。
        valid_mat: 验证集用户-物品交互矩阵。
        test_mat: 测试集用户-物品交互矩阵。

    返回:
        eval_results: 包含验证集和测试集上P@K, R@K, N@K评价指标结果的字典。
    """

    # 初始化评价指标字典，包含不同截止点（10, 20, 50）的Precision (P), Recall (R) 和 NDCG (N)
    metrics = {'P10': [], 'P20': [], 'P50': [], 'R10': [], 'R20': [], 'R50': [], 'N10': [], 'N20': [], 'N50': []}

    # 创建一个字典来存储验证集和测试集的评价结果，每个键对应于metrics的一个深拷贝
    eval_results = {'valid': copy.deepcopy(metrics), 'test': copy.deepcopy(metrics)}

    # 获取用户和物品的在线（online）和目标（target）嵌入向量
    u_online, u_target, i_online, i_target = model.get_embedding()

    # 计算用户-物品评分矩阵，分别表示从用户到物品和从物品到用户的预测偏好
    score_mat_ui = torch.matmul(u_online, i_target.transpose(0, 1))
    score_mat_iu = torch.matmul(u_target, i_online.transpose(0, 1))

    # 将两个评分矩阵相加得到最终的评分矩阵
    score_mat = score_mat_ui + score_mat_iu

    # 对评分矩阵按行进行降序排序，获得每个用户对所有物品的偏好排序列表
    sorted_mat = torch.argsort(score_mat.cpu(), dim=1, descending=True)

    # 遍历测试集中的每个用户
    for test_user in test_mat:
        # 提取该用户对所有物品的排序列表，并将其转换为Python列表
        sorted_list = list(to_np(sorted_mat[test_user]))

        # 分别处理验证集和测试集
        for mode in ['valid', 'test']:
            sorted_list_tmp = []

            # 根据模式选择合适的ground truth矩阵和已经看过的物品集合
            if mode == 'valid':
                gt_mat = valid_mat
                already_seen_items = set(train_mat[test_user].keys()) | set(test_mat[test_user].keys())
            elif mode == 'test':
                gt_mat = test_mat
                already_seen_items = set(train_mat[test_user].keys()) | set(valid_mat[test_user].keys())

            # 筛选出新的推荐物品，直到达到设定的数量（这里是50个）
            for item in sorted_list:
                if item not in already_seen_items:
                    sorted_list_tmp.append(item)
                if len(sorted_list_tmp) == 50: break

            # 计算命中次数，即前K项中有多少是用户实际喜欢的物品
            hit_10 = len(set(sorted_list_tmp[:10]) & set(gt_mat[test_user].keys()))
            hit_20 = len(set(sorted_list_tmp[:20]) & set(gt_mat[test_user].keys()))
            hit_50 = len(set(sorted_list_tmp[:50]) & set(gt_mat[test_user].keys()))

            # 计算并保存Precision@K, Recall@K
            eval_results[mode]['P10'].append(hit_10 / min(10, len(gt_mat[test_user].keys())))
            eval_results[mode]['P20'].append(hit_20 / min(20, len(gt_mat[test_user].keys())))
            eval_results[mode]['P50'].append(hit_50 / min(50, len(gt_mat[test_user].keys())))

            eval_results[mode]['R10'].append(hit_10 / len(gt_mat[test_user].keys()))
            eval_results[mode]['R20'].append(hit_20 / len(gt_mat[test_user].keys()))
            eval_results[mode]['R50'].append(hit_50 / len(gt_mat[test_user].keys()))

            # 计算并保存NDCG@K
            denom = np.log2(np.arange(2, 10 + 2))  # 计算分母，即位置折扣
            dcg_10 = np.sum(np.in1d(sorted_list_tmp[:10], list(gt_mat[test_user].keys())) / denom)  # 计算DCG
            idcg_10 = np.sum((1 / denom)[:min(len(list(gt_mat[test_user].keys())), 10)])  # 计算IDCG

            denom = np.log2(np.arange(2, 20 + 2))
            dcg_20 = np.sum(np.in1d(sorted_list_tmp[:20], list(gt_mat[test_user].keys())) / denom)
            idcg_20 = np.sum((1 / denom)[:min(len(list(gt_mat[test_user].keys())), 20)])

            denom = np.log2(np.arange(2, 50 + 2))
            dcg_50 = np.sum(np.in1d(sorted_list_tmp[:50], list(gt_mat[test_user].keys())) / denom)
            idcg_50 = np.sum((1 / denom)[:min(len(list(gt_mat[test_user].keys())), 50)])

            eval_results[mode]['N10'].append(dcg_10 / idcg_10)  # 计算NDCG
            eval_results[mode]['N20'].append(dcg_20 / idcg_20)
            eval_results[mode]['N50'].append(dcg_50 / idcg_50)

    # 计算并保存每种模式下各个截止点的平均P, R, N值
    for mode in ['test', 'valid']:
        for topk in [10, 20, 50]:
            eval_results[mode]['P' + str(topk)] = round(np.asarray(eval_results[mode]['P' + str(topk)]).mean(), 4)
            eval_results[mode]['R' + str(topk)] = round(np.asarray(eval_results[mode]['R' + str(topk)]).mean(), 4)
            eval_results[mode]['N' + str(topk)] = round(np.asarray(eval_results[mode]['N' + str(topk)]).mean(), 4)

    return eval_results


def print_eval_results(eval_results):
    """
    打印评估结果，包括验证集和测试集上的Precision, Recall, NDCG指标。

    参数:
        eval_results: 由evaluate函数返回的评价结果字典。
    """

    # 遍历验证集和测试集的结果
    for mode in ['valid', 'test']:
        for topk in [10, 20, 50]:
            p = eval_results[mode]['P' + str(topk)]
            r = eval_results[mode]['R' + str(topk)]
            n = eval_results[mode]['N' + str(topk)]

            # 使用字符串格式化方法美化输出
            print('{:5s} P@{}: {:.4f}, R@{}: {:.4f}, N@{}: {:.4f}'.format(mode.upper(), topk, p, topk, r, topk, n))
        print()  # 在验证集和测试集之间打印空行以区分