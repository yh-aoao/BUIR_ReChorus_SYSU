import torch
import torch.utils.data as data



class ImplicitFeedback(data.Dataset):
    """
    这段代码定义了一个名为 ImplicitFeedback 的自定义数据集类，它继承自 PyTorch 的 Dataset 类。
    该类用于处理隐式反馈（implicit feedback）数据，通常在推荐系统中使用。
    隐式反馈是指用户行为数据，如点击、购买或浏览历史，而不是显式的评分。
    """
    def __init__(self, user_count, item_count, interaction_mat):
        """
        初始化函数，设置数据集的基本参数。

        参数:
            user_count (int): 用户总数。
            item_count (int): 商品总数。
            interaction_mat (dict): 交互矩阵，字典类型，键是用户ID，值是该用户有过交互的商品ID列表。
        """
        super(ImplicitFeedback, self).__init__()

        self.user_count = user_count  # 记录用户的数量
        self.item_count = item_count  # 记录商品的数量

        self.interactions = []  # 用来存储所有的用户-商品交互对
        for user in interaction_mat:  # 遍历所有用户
            for item in interaction_mat[user]:  # 遍历该用户与之交互的所有商品
                # 将每个用户-商品交互对作为一条记录添加到列表中，标记为1表示有交互
                self.interactions.append([user, item, 1])

    def __len__(self):
        """
        返回数据集中样本的数量。

        返回:
            int: 数据集中样本的总数。
        """
        return len(self.interactions)

    def __getitem__(self, idx):
        """
        根据索引获取单个数据点。

        参数:
            idx (int): 要获取的数据点的索引。

        返回:
            dict: 包含'user'和'item'两个键的字典，分别对应用户ID和商品ID。
        """
        return {
            'user': self.interactions[idx][0],  # 用户ID
            'item': self.interactions[idx][1],  # 商品ID
            # 注意：这里没有返回交互标志（即第三个元素），如果需要可以在__getitem__方法中添加
        }

