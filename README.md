# BUIR_ReChorus_SYSU

## Sun Yat-sen University Artificial Intelligence College Machine Learning Course Project

## 安装
1.克隆项目
git clone https://github.com/yh-aoao/BUIR_ReChorus_SYSU.git

2.安装依赖
每个模型里面都有相应的依赖文件，进入对应的模型里输入下列命令行（注：BUIR_ReChorus模型和其他模型依赖不同）

pip install -r requirements.txt

## 模型说明
1.BUIR模型

来源：https://github.com/donalee/BUIR

BUIR: Bootstrapping User and Item Representations for One-Class Collaborative Filtering

BUIR模型通过分析四个核心要素：行为（Behavior）、用户（User）、信息（Information）和结果（Result），帮助理解和优化用户与系统的互动过程。它关注用户的需求和行为，确保信息有效传达，并最终提高用户体验和任务完成效果。

2.BUIR_ReChorus模型

我们用Rechorus框架复现了BUIR_NB模型，放在了BUIR_ReChorus/src/models/general/BUIR_NB

ReChours是一种基于深度学习的推荐系统模型，

3.BUIR_PYG模型

我们在BUIR模型的基础上进行了改进，原BUIR_NB的图卷积是通过自定义实现的，我们改用PYG库来实现图卷积功能，发现效果比原模型更好

该模型放在了BUIR_PYG/Models/BUIR_NB

使用命令行python main.py --dataset toy-dataset --model buir-nb运行

（注意：因为作者们并没有GPU，所以全部代码都使用CPU运行）

4.BUIR_GAT模型

我们在原BUIR_NB模型的基础上，我们还引入了注意力机制（GAT），使用GAT模型来进行改进

该模型放在了BUIR_GAT/Models/BUIR_NB

使用命令行python main.py --dataset toy-dataset --model buir-nb运行

5.BUIR_GCN模型

我们在原BUIR_NB模型的基础上，我们还引入图神经网络的解释性，使用GCN模型来进行改进

该模型放在了放在了BUIR_GAT/Models/BUIR_NB

使用命令行python main.py --dataset toy-dataset --model buir-nb运行

## 实验
1.复现实验

我们在ReChorus框架上复现了BUIR_NB模型，并使用ReChorus框架里的三个数据集“Grocery_and_Gourmet_Food”、“MIND_Large”、“MovieLens_1M”来与原ReChorus中同类别的两个其他模型（这里说明哪两个模型）进行对比

2.改进实验

我们在原BUIR_NB模型的基础上进行了改进，具体改进有：用PYG库来实现图卷积功能、引入了注意力机制（GAT）、引入图神经网络的解释性

并在原BUIR_NB模型的数据集“toy-dataset”上进行了对比实验

## 实验结果
1.复现实验结果

2.改进实验结果