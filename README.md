# BUIR_ReChorus_SYSU
Sun Yat-sen University Artificial Intelligence College Machine Learning Course Project

1.我们用Rechorus复现了BUIR_NB模型，放在了BUIR_ReChorus/src/models/general/BUIR_NB
使用命令行python .\src\main.py --model_name BUIR_NB --emb_size 64 --lr 1e-3 --l2 1e-6 --dataset 'Grocery_and_Gourmet_Food'允许


2.我们在BUIR模型的基础上进行了改进，原BUIR_NB的图卷积是通过自定义实现的，我们改用PYG库来实现图卷积功能，发现效果比原模型更好
放在了BUIR_PYG/Models/BUIR_NB
使用命令行python main.py --dataset toy-dataset --model buir-nb运行
注意，因为作者们并没有GPU，所以全部代码都使用CPU运行

3.并且我们还引入了注意力机制（GAT）
放在了BUIR_GAT/Models/BUIR_NB
使用命令行python main.py --dataset toy-dataset --model buir-nb运行

4.引入图神经网络的解释性
放在了BUIR_GCN/Models/BUIR_NB
使用命令行python main.py --dataset toy-dataset --model buir-nb运行
