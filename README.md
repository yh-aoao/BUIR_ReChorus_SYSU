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
### 1.复现实验

我们在ReChorus框架上复现了BUIR_NB模型，并使用ReChorus框架里的三个数据集“Grocery_and_Gourmet_Food”、“MIND_Large”来与原ReChorus中同类别的两个其他模型（BRRMF、NEUMF）进行对比

### 2.改进实验

我们在原BUIR_NB模型的基础上进行了改进，具体改进有：用PYG库来实现图卷积功能、引入了注意力机制（GAT）、引入图神经网络的解释性

并在原BUIR_NB模型的数据集“toy-dataset”上进行了对比实验

## 实验结果
### 1.复现实验结果
#### 1.1数据集“Grocery_and_Gourmet_Food”
##### 1.1.1 BUIR_NB模型结果
```plaintext
Test Before Training: (HR@5:0.0000,NDCG@5:0.0000,HR@10:0.0000,NDCG@10:0.0000,HR@20:0.0000,NDCG@20:0.0000,HR@50:0.0000,NDCG@50:0.0000)
Optimizer: Adam
Epoch 1     loss=0.7689 [233.5 s]       dev=(HR@5:0.0029,NDCG@5:0.0015) [77.9 s] *
Epoch 2     loss=0.0371 [206.5 s]       dev=(HR@5:0.0029,NDCG@5:0.0015) [72.3 s] *
Epoch 3     loss=0.0213 [205.6 s]       dev=(HR@5:0.0029,NDCG@5:0.0016) [69.4 s] *
Epoch 4     loss=0.0170 [206.3 s]       dev=(HR@5:0.0029,NDCG@5:0.0015) [68.7 s]
Epoch 5     loss=0.0173 [206.5 s]       dev=(HR@5:0.0029,NDCG@5:0.0015) [81.0 s]
Epoch 6     loss=0.0203 [250.3 s]       dev=(HR@5:0.0031,NDCG@5:0.0016) [70.3 s] *
Epoch 7     loss=0.0247 [208.2 s]       dev=(HR@5:0.0030,NDCG@5:0.0016) [71.6 s]
Epoch 8     loss=0.0303 [229.4 s]       dev=(HR@5:0.0034,NDCG@5:0.0017) [80.9 s] *
Epoch 9     loss=0.0382 [227.3 s]       dev=(HR@5:0.0035,NDCG@5:0.0018) [69.5 s] *
Epoch 10    loss=0.0490 [207.8 s]       dev=(HR@5:0.0032,NDCG@5:0.0016) [75.3 s]
Epoch 11    loss=0.0626 [196.6 s]       dev=(HR@5:0.0015,NDCG@5:0.0007) [68.7 s]
Epoch 12    loss=0.0752 [206.3 s]       dev=(HR@5:0.0016,NDCG@5:0.0009) [73.9 s]
Epoch 13    loss=0.0732 [223.9 s]       dev=(HR@5:0.0016,NDCG@5:0.0009) [77.4 s]
Epoch 14    loss=0.0487 [294.8 s]       dev=(HR@5:0.0021,NDCG@5:0.0009) [90.5 s]
Epoch 15    loss=0.0300 [293.0 s]       dev=(HR@5:0.0030,NDCG@5:0.0015) [114.9 s]
Epoch 16    loss=0.0100 [508.4 s]       dev=(HR@5:0.0031,NDCG@5:0.0016) [110.2 s]
Epoch 17    loss=0.0062 [335.3 s]       dev=(HR@5:0.0031,NDCG@5:0.0016) [104.0 s]
Epoch 18    loss=0.0061 [318.1 s]       dev=(HR@5:0.0028,NDCG@5:0.0014) [100.7 s]
Early stop at 18 based on dev result.
Best Iter(dev)=    9     dev=(HR@5:0.0035,NDCG@5:0.0018) [6035.2 s]
```
##### 1.1.2 BRRMF模型结果
```plaintext
INFO:root:Test Before Training: (HR@5:0.0523,NDCG@5:0.0313,HR@10:0.1034,NDCG@10:0.0475,HR@20:0.2037,NDCG@20:0.0726,HR@50:0.5053,NDCG@50:0.1314)
INFO:root:Optimizer: Adam
INFO:root:Epoch 1     loss=0.6690 [23.5 s]	dev=(HR@5:0.2094,NDCG@5:0.1341) [20.0 s] *
INFO:root:Epoch 2     loss=0.5213 [19.3 s]	dev=(HR@5:0.2411,NDCG@5:0.1558) [31.2 s] *
INFO:root:Epoch 3     loss=0.4521 [34.4 s]	dev=(HR@5:0.2579,NDCG@5:0.1716) [17.6 s] *
INFO:root:Epoch 4     loss=0.4076 [24.1 s]	dev=(HR@5:0.2772,NDCG@5:0.1893) [18.9 s] *
INFO:root:Epoch 5     loss=0.3687 [21.6 s]	dev=(HR@5:0.2951,NDCG@5:0.2028) [18.7 s] *
INFO:root:Epoch 6     loss=0.3308 [21.9 s]	dev=(HR@5:0.3069,NDCG@5:0.2142) [18.8 s] *
INFO:root:Epoch 7     loss=0.2947 [21.6 s]	dev=(HR@5:0.3184,NDCG@5:0.2234) [18.5 s] *
INFO:root:Epoch 8     loss=0.2623 [25.7 s]	dev=(HR@5:0.3314,NDCG@5:0.2333) [32.0 s] *
INFO:root:Epoch 9     loss=0.2315 [20.6 s]	dev=(HR@5:0.3411,NDCG@5:0.2418) [22.3 s] *
INFO:root:Epoch 10    loss=0.2061 [24.0 s]	dev=(HR@5:0.3480,NDCG@5:0.2485) [22.9 s] *
INFO:root:Epoch 11    loss=0.1813 [23.9 s]	dev=(HR@5:0.3539,NDCG@5:0.2531) [22.5 s] *
INFO:root:Epoch 12    loss=0.1600 [24.1 s]	dev=(HR@5:0.3585,NDCG@5:0.2571) [23.5 s] *
INFO:root:Epoch 13    loss=0.1419 [24.1 s]	dev=(HR@5:0.3648,NDCG@5:0.2617) [19.5 s] *
INFO:root:Epoch 14    loss=0.1268 [36.0 s]	dev=(HR@5:0.3696,NDCG@5:0.2646) [28.4 s] *
INFO:root:Epoch 15    loss=0.1116 [21.6 s]	dev=(HR@5:0.3728,NDCG@5:0.2666) [21.2 s] *
INFO:root:Epoch 16    loss=0.1016 [25.9 s]	dev=(HR@5:0.3768,NDCG@5:0.2691) [20.4 s] *
INFO:root:Epoch 17    loss=0.0913 [25.8 s]	dev=(HR@5:0.3788,NDCG@5:0.2712) [20.6 s] *
INFO:root:Epoch 18    loss=0.0837 [26.2 s]	dev=(HR@5:0.3804,NDCG@5:0.2726) [20.3 s] *
INFO:root:Epoch 19    loss=0.0764 [25.4 s]	dev=(HR@5:0.3831,NDCG@5:0.2745) [19.1 s] *
INFO:root:Epoch 20    loss=0.0696 [41.2 s]	dev=(HR@5:0.3866,NDCG@5:0.2767) [18.1 s] *
INFO:root:Epoch 21    loss=0.0637 [25.6 s]	dev=(HR@5:0.3876,NDCG@5:0.2780) [20.9 s] *
INFO:root:Epoch 22    loss=0.0597 [29.6 s]	dev=(HR@5:0.3876,NDCG@5:0.2782) [21.6 s] *
INFO:root:Epoch 23    loss=0.0559 [25.6 s]	dev=(HR@5:0.3878,NDCG@5:0.2788) [21.3 s] *
INFO:root:Epoch 24    loss=0.0523 [23.9 s]	dev=(HR@5:0.3905,NDCG@5:0.2803) [23.4 s] *
INFO:root:Epoch 25    loss=0.0499 [21.2 s]	dev=(HR@5:0.3903,NDCG@5:0.2791) [36.8 s]
INFO:root:Epoch 26    loss=0.0476 [37.8 s]	dev=(HR@5:0.3905,NDCG@5:0.2796) [24.3 s]
INFO:root:Epoch 27    loss=0.0460 [40.5 s]	dev=(HR@5:0.3908,NDCG@5:0.2807) [23.5 s] *
INFO:root:Epoch 28    loss=0.0441 [24.6 s]	dev=(HR@5:0.3928,NDCG@5:0.2819) [24.2 s] *
INFO:root:Epoch 29    loss=0.0422 [25.4 s]	dev=(HR@5:0.3926,NDCG@5:0.2823) [23.0 s] *
INFO:root:Epoch 30    loss=0.0409 [24.6 s]	dev=(HR@5:0.3940,NDCG@5:0.2832) [21.4 s] *
INFO:root:Epoch 31    loss=0.0395 [27.5 s]	dev=(HR@5:0.3965,NDCG@5:0.2848) [33.0 s] *
INFO:root:Epoch 32    loss=0.0388 [23.8 s]	dev=(HR@5:0.3967,NDCG@5:0.2854) [20.3 s] *
INFO:root:Epoch 33    loss=0.0379 [26.1 s]	dev=(HR@5:0.3983,NDCG@5:0.2865) [20.8 s] *
INFO:root:Epoch 34    loss=0.0367 [27.1 s]	dev=(HR@5:0.3993,NDCG@5:0.2873) [20.7 s] *
INFO:root:Epoch 35    loss=0.0359 [26.3 s]	dev=(HR@5:0.3985,NDCG@5:0.2866) [21.0 s]
INFO:root:Epoch 36    loss=0.0361 [26.5 s]	dev=(HR@5:0.3990,NDCG@5:0.2877) [17.1 s] *
INFO:root:Epoch 37    loss=0.0348 [40.2 s]	dev=(HR@5:0.4003,NDCG@5:0.2886) [24.0 s] *
INFO:root:Epoch 38    loss=0.0350 [24.4 s]	dev=(HR@5:0.4015,NDCG@5:0.2900) [20.7 s] *
INFO:root:Epoch 39    loss=0.0340 [26.0 s]	dev=(HR@5:0.4013,NDCG@5:0.2897) [21.4 s]
INFO:root:Epoch 40    loss=0.0338 [24.1 s]	dev=(HR@5:0.3994,NDCG@5:0.2884) [22.9 s]
INFO:root:Epoch 41    loss=0.0333 [24.2 s]	dev=(HR@5:0.4009,NDCG@5:0.2886) [23.0 s]
INFO:root:Epoch 42    loss=0.0335 [22.4 s]	dev=(HR@5:0.4017,NDCG@5:0.2895) [25.0 s]
INFO:root:Epoch 43    loss=0.0329 [38.3 s]	dev=(HR@5:0.4035,NDCG@5:0.2900) [18.0 s] *
INFO:root:Epoch 44    loss=0.0328 [25.3 s]	dev=(HR@5:0.4047,NDCG@5:0.2913) [23.1 s] *
INFO:root:Epoch 45    loss=0.0320 [24.0 s]	dev=(HR@5:0.4056,NDCG@5:0.2923) [23.0 s] *
INFO:root:Epoch 46    loss=0.0324 [24.3 s]	dev=(HR@5:0.4032,NDCG@5:0.2921) [22.1 s]
INFO:root:Epoch 47    loss=0.0314 [25.4 s]	dev=(HR@5:0.4051,NDCG@5:0.2928) [23.0 s] *
INFO:root:Epoch 48    loss=0.0312 [25.8 s]	dev=(HR@5:0.4070,NDCG@5:0.2933) [31.5 s] *
INFO:root:Epoch 49    loss=0.0314 [37.6 s]	dev=(HR@5:0.4064,NDCG@5:0.2939) [18.9 s] *
INFO:root:Epoch 50    loss=0.0309 [27.8 s]	dev=(HR@5:0.4077,NDCG@5:0.2946) [20.8 s] *
INFO:root:Epoch 51    loss=0.0314 [26.0 s]	dev=(HR@5:0.4060,NDCG@5:0.2934) [21.0 s]
INFO:root:Epoch 52    loss=0.0305 [29.6 s]	dev=(HR@5:0.4070,NDCG@5:0.2940) [21.0 s]
INFO:root:Epoch 53    loss=0.0306 [28.5 s]	dev=(HR@5:0.4064,NDCG@5:0.2935) [16.8 s]
INFO:root:Epoch 54    loss=0.0301 [42.8 s]	dev=(HR@5:0.4077,NDCG@5:0.2940) [31.0 s]
INFO:root:Epoch 55    loss=0.0305 [25.0 s]	dev=(HR@5:0.4084,NDCG@5:0.2951) [26.7 s] *
INFO:root:Epoch 56    loss=0.0298 [27.8 s]	dev=(HR@5:0.4107,NDCG@5:0.2963) [24.0 s] *
INFO:root:Epoch 57    loss=0.0298 [29.8 s]	dev=(HR@5:0.4123,NDCG@5:0.2972) [22.9 s] *
INFO:root:Epoch 58    loss=0.0299 [24.1 s]	dev=(HR@5:0.4116,NDCG@5:0.2969) [23.3 s]
INFO:root:Epoch 59    loss=0.0299 [25.2 s]	dev=(HR@5:0.4110,NDCG@5:0.2964) [24.4 s]
INFO:root:Epoch 60    loss=0.0292 [43.2 s]	dev=(HR@5:0.4109,NDCG@5:0.2966) [20.0 s]
INFO:root:Epoch 61    loss=0.0300 [23.3 s]	dev=(HR@5:0.4111,NDCG@5:0.2964) [22.0 s]
INFO:root:Epoch 62    loss=0.0292 [24.9 s]	dev=(HR@5:0.4108,NDCG@5:0.2969) [20.9 s]
INFO:root:Epoch 63    loss=0.0287 [26.0 s]	dev=(HR@5:0.4101,NDCG@5:0.2971) [20.7 s]
INFO:root:Epoch 64    loss=0.0285 [26.2 s]	dev=(HR@5:0.4105,NDCG@5:0.2978) [20.8 s] *
INFO:root:Epoch 65    loss=0.0290 [23.4 s]	dev=(HR@5:0.4117,NDCG@5:0.2991) [29.5 s] *
INFO:root:Epoch 66    loss=0.0289 [33.3 s]	dev=(HR@5:0.4107,NDCG@5:0.2989) [17.2 s]
INFO:root:Epoch 67    loss=0.0285 [26.1 s]	dev=(HR@5:0.4103,NDCG@5:0.2984) [20.3 s]
INFO:root:Epoch 68    loss=0.0283 [26.9 s]	dev=(HR@5:0.4113,NDCG@5:0.2988) [20.5 s]
INFO:root:Epoch 69    loss=0.0283 [26.5 s]	dev=(HR@5:0.4116,NDCG@5:0.2989) [20.5 s]
INFO:root:Epoch 70    loss=0.0283 [24.5 s]	dev=(HR@5:0.4127,NDCG@5:0.2995) [20.9 s] *
INFO:root:Epoch 71    loss=0.0280 [27.8 s]	dev=(HR@5:0.4135,NDCG@5:0.2999) [35.4 s] *
INFO:root:Epoch 72    loss=0.0282 [18.6 s]	dev=(HR@5:0.4126,NDCG@5:0.2984) [22.8 s]
INFO:root:Epoch 73    loss=0.0283 [23.9 s]	dev=(HR@5:0.4113,NDCG@5:0.2982) [22.5 s]
INFO:root:Epoch 74    loss=0.0280 [23.8 s]	dev=(HR@5:0.4109,NDCG@5:0.2986) [22.9 s]
INFO:root:Epoch 75    loss=0.0276 [24.0 s]	dev=(HR@5:0.4131,NDCG@5:0.2997) [22.8 s]
INFO:root:Epoch 76    loss=0.0278 [23.8 s]	dev=(HR@5:0.4139,NDCG@5:0.2999) [18.1 s] *
INFO:root:Epoch 77    loss=0.0276 [37.9 s]	dev=(HR@5:0.4139,NDCG@5:0.2999) [23.8 s] *
INFO:root:Epoch 78    loss=0.0277 [22.2 s]	dev=(HR@5:0.4118,NDCG@5:0.2996) [20.7 s]
INFO:root:Epoch 79    loss=0.0271 [26.3 s]	dev=(HR@5:0.4122,NDCG@5:0.2993) [20.5 s]
INFO:root:Epoch 80    loss=0.0279 [26.1 s]	dev=(HR@5:0.4140,NDCG@5:0.3002) [20.6 s] *
INFO:root:Epoch 81    loss=0.0275 [25.9 s]	dev=(HR@5:0.4151,NDCG@5:0.3006) [20.7 s] *
INFO:root:Epoch 82    loss=0.0269 [24.3 s]	dev=(HR@5:0.4128,NDCG@5:0.2998) [31.3 s]
INFO:root:Epoch 83    loss=0.0272 [41.4 s]	dev=(HR@5:0.4116,NDCG@5:0.2991) [21.5 s]
INFO:root:Epoch 84    loss=0.0278 [36.5 s]	dev=(HR@5:0.4122,NDCG@5:0.2999) [35.1 s]
INFO:root:Epoch 85    loss=0.0271 [33.4 s]	dev=(HR@5:0.4123,NDCG@5:0.3002) [21.2 s]
INFO:root:Epoch 86    loss=0.0271 [26.1 s]	dev=(HR@5:0.4125,NDCG@5:0.3000) [23.5 s]
INFO:root:Epoch 87    loss=0.0271 [24.5 s]	dev=(HR@5:0.4135,NDCG@5:0.3004) [24.1 s]
INFO:root:Epoch 88    loss=0.0270 [19.8 s]	dev=(HR@5:0.4118,NDCG@5:0.2993) [35.2 s]
INFO:root:Epoch 89    loss=0.0268 [27.6 s]	dev=(HR@5:0.4115,NDCG@5:0.3003) [20.5 s]
INFO:root:Epoch 90    loss=0.0270 [24.1 s]	dev=(HR@5:0.4146,NDCG@5:0.3016) [22.8 s] *
INFO:root:Epoch 91    loss=0.0266 [24.2 s]	dev=(HR@5:0.4150,NDCG@5:0.3022) [22.9 s] *
INFO:root:Epoch 92    loss=0.0270 [24.0 s]	dev=(HR@5:0.4154,NDCG@5:0.3013) [21.9 s]
INFO:root:Epoch 93    loss=0.0269 [24.7 s]	dev=(HR@5:0.4126,NDCG@5:0.3002) [19.3 s]
INFO:root:Epoch 94    loss=0.0266 [30.9 s]	dev=(HR@5:0.4140,NDCG@5:0.3014) [31.6 s]
INFO:root:Epoch 95    loss=0.0269 [21.7 s]	dev=(HR@5:0.4135,NDCG@5:0.3012) [25.1 s]
INFO:root:Epoch 96    loss=0.0263 [27.5 s]	dev=(HR@5:0.4142,NDCG@5:0.3013) [24.5 s]
INFO:root:Epoch 97    loss=0.0262 [27.7 s]	dev=(HR@5:0.4139,NDCG@5:0.3013) [22.3 s]
INFO:root:Epoch 98    loss=0.0264 [28.8 s]	dev=(HR@5:0.4128,NDCG@5:0.3009) [22.1 s]
INFO:root:Epoch 99    loss=0.0268 [28.4 s]	dev=(HR@5:0.4140,NDCG@5:0.3015) [19.8 s]
INFO:root:Epoch 100   loss=0.0262 [50.9 s]	dev=(HR@5:0.4147,NDCG@5:0.3019) [26.4 s]
INFO:root:Early stop at 100 based on dev result.
INFO:root:
Best Iter(dev)=   91	 dev=(HR@5:0.4150,NDCG@5:0.3022) [5018.3 s]
```
##### 1.1.3 NEUMF模型结果
```plaintext
INFO:root:Test Before Training: (HR@5:0.0524,NDCG@5:0.0301,HR@10:0.1010,NDCG@10:0.0456,HR@20:0.1978,NDCG@20:0.0698,HR@50:0.4901,NDCG@50:0.1269)
INFO:root:Optimizer: Adam
INFO:root:Epoch 1     loss=0.5065 [31.6 s]	dev=(HR@5:0.2417,NDCG@5:0.1584) [19.8 s] *
INFO:root:Epoch 2     loss=0.4459 [31.0 s]	dev=(HR@5:0.2421,NDCG@5:0.1629) [22.3 s] *
INFO:root:Epoch 3     loss=0.4183 [31.1 s]	dev=(HR@5:0.2706,NDCG@5:0.1816) [26.5 s] *
INFO:root:Epoch 4     loss=0.3852 [35.5 s]	dev=(HR@5:0.3022,NDCG@5:0.2080) [27.6 s] *
INFO:root:Epoch 5     loss=0.3471 [38.1 s]	dev=(HR@5:0.3173,NDCG@5:0.2173) [28.2 s] *
INFO:root:Epoch 6     loss=0.3119 [31.6 s]	dev=(HR@5:0.3292,NDCG@5:0.2261) [28.7 s] *
INFO:root:Epoch 7     loss=0.2771 [40.0 s]	dev=(HR@5:0.3331,NDCG@5:0.2289) [24.7 s] *
INFO:root:Epoch 8     loss=0.2472 [39.3 s]	dev=(HR@5:0.3361,NDCG@5:0.2314) [24.6 s] *
INFO:root:Epoch 9     loss=0.2199 [36.3 s]	dev=(HR@5:0.3388,NDCG@5:0.2330) [24.2 s] *
INFO:root:Epoch 10    loss=0.1956 [37.0 s]	dev=(HR@5:0.3394,NDCG@5:0.2353) [27.6 s] *
INFO:root:Epoch 11    loss=0.1770 [34.0 s]	dev=(HR@5:0.3389,NDCG@5:0.2333) [26.7 s]
INFO:root:Epoch 12    loss=0.1608 [29.9 s]	dev=(HR@5:0.3377,NDCG@5:0.2332) [27.7 s]
INFO:root:Epoch 13    loss=0.1464 [35.5 s]	dev=(HR@5:0.3398,NDCG@5:0.2362) [25.3 s] *
INFO:root:Epoch 14    loss=0.1348 [36.2 s]	dev=(HR@5:0.3396,NDCG@5:0.2351) [24.5 s]
INFO:root:Epoch 15    loss=0.1235 [35.6 s]	dev=(HR@5:0.3387,NDCG@5:0.2345) [24.4 s]
INFO:root:Epoch 16    loss=0.1159 [36.0 s]	dev=(HR@5:0.3362,NDCG@5:0.2319) [25.0 s]
INFO:root:Epoch 17    loss=0.1099 [36.3 s]	dev=(HR@5:0.3354,NDCG@5:0.2313) [27.1 s]
INFO:root:Epoch 18    loss=0.1031 [31.0 s]	dev=(HR@5:0.3362,NDCG@5:0.2317) [27.9 s]
INFO:root:Epoch 19    loss=0.0989 [31.0 s]	dev=(HR@5:0.3342,NDCG@5:0.2305) [27.9 s]
INFO:root:Epoch 20    loss=0.0944 [32.9 s]	dev=(HR@5:0.3353,NDCG@5:0.2311) [28.3 s]
INFO:root:Epoch 21    loss=0.0891 [36.8 s]	dev=(HR@5:0.3360,NDCG@5:0.2315) [24.3 s]
INFO:root:Epoch 22    loss=0.0873 [31.7 s]	dev=(HR@5:0.3349,NDCG@5:0.2316) [23.9 s]
INFO:root:Early stop at 22 based on dev result.
INFO:root:
Best Iter(dev)=   13	 dev=(HR@5:0.3398,NDCG@5:0.2362) [1325.6 s] 
```

#### 1.2 数据集“MIND_Large”
##### 1.2.1 BUIR_NB模型结果
```plaintext
INFO:root:Test Before Training: (HR@5:0.0000,NDCG@5:0.0000,HR@10:0.0000,NDCG@10:0.0000,HR@20:0.0000,NDCG@20:0.0000,HR@50:0.0000,NDCG@50:0.0000)
INFO:root:Optimizer: Adam
INFO:root:Epoch 1     loss=0.7068 [273.1 s]       dev=(HR@5:0.0120,NDCG@5:0.0060) [28.1 s] *
INFO:root:Epoch 2     loss=0.0380 [276.0 s]       dev=(HR@5:0.0240,NDCG@5:0.0120) [27.9 s] *
INFO:root:Epoch 3     loss=0.0200 [274.5 s]       dev=(HR@5:0.0360,NDCG@5:0.0180) [25.2 s] *
INFO:root:Epoch 4     loss=0.0131 [318.4 s]       dev=(HR@5:0.0480,NDCG@5:0.0240) [36.3 s] *
INFO:root:Epoch 5     loss=0.0095 [338.4 s]       dev=(HR@5:0.0600,NDCG@5:0.0300) [34.7 s] *
INFO:root:Epoch 6     loss=0.0072 [338.8 s]       dev=(HR@5:0.0720,NDCG@5:0.0360) [31.7 s] *
INFO:root:Epoch 7     loss=0.0056 [317.5 s]       dev=(HR@5:0.0840,NDCG@5:0.0420) [27.1 s] *
INFO:root:Epoch 8     loss=0.0050 [236.4 s]       dev=(HR@5:0.0960,NDCG@5:0.0480) [23.1 s] *
INFO:root:Epoch 9     loss=0.0051 [232.6 s]       dev=(HR@5:0.1080,NDCG@5:0.0540) [23.0 s] 
INFO:root:Epoch 10    loss=0.0047 [232.6 s]       dev=(HR@5:0.1200,NDCG@5:0.0600) [23.0 s] *
INFO:root:Epoch 11    loss=0.0093 [225.0 s]       dev=(HR@5:0.1320,NDCG@5:0.0660) [21.8 s] *
INFO:root:Early stop at 11 based on dev result.
INFO:root:
Best Iter(dev)=    10	 dev=(HR@5:0.1200,NDCG@5:0.0600) [3365.3 s]
```
##### 1.2.2 BRRMF模型结果
```plaintext
INFO:root:Test Before Training: (HR@5:0.0441,NDCG@5:0.0265,HR@10:0.1000,NDCG@10:0.0446,HR@20:0.1853,NDCG@20:0.0659,HR@50:0.4990,NDCG@50:0.1271)
INFO:root:Optimizer: Adam
INFO:root:Epoch 1     loss=0.6541 [25.2 s]	dev=(HR@5:0.0371,NDCG@5:0.0209) [18.5 s] *
INFO:root:Epoch 2     loss=0.4194 [26.6 s]	dev=(HR@5:0.0337,NDCG@5:0.0199) [18.8 s]
INFO:root:Epoch 3     loss=0.3537 [26.6 s]	dev=(HR@5:0.0334,NDCG@5:0.0201) [18.7 s]
INFO:root:Epoch 4     loss=0.3371 [26.6 s]	dev=(HR@5:0.0383,NDCG@5:0.0227) [18.5 s] *
INFO:root:Epoch 5     loss=0.3220 [27.1 s]	dev=(HR@5:0.0377,NDCG@5:0.0224) [19.1 s]
INFO:root:Epoch 6     loss=0.3097 [28.6 s]	dev=(HR@5:0.0399,NDCG@5:0.0230) [15.0 s] *
INFO:root:Epoch 7     loss=0.2952 [25.3 s]	dev=(HR@5:0.0448,NDCG@5:0.0257) [20.1 s] *
INFO:root:Epoch 8     loss=0.2789 [23.6 s]	dev=(HR@5:0.0439,NDCG@5:0.0250) [22.1 s]
INFO:root:Epoch 9     loss=0.2630 [23.2 s]	dev=(HR@5:0.0469,NDCG@5:0.0260) [20.9 s] *
INFO:root:Epoch 10    loss=0.2464 [23.6 s]	dev=(HR@5:0.0503,NDCG@5:0.0284) [20.8 s] *
INFO:root:Epoch 11    loss=0.2270 [23.0 s]	dev=(HR@5:0.0525,NDCG@5:0.0297) [20.9 s] *
INFO:root:Epoch 12    loss=0.2096 [22.8 s]	dev=(HR@5:0.0485,NDCG@5:0.0279) [19.8 s]
INFO:root:Epoch 13    loss=0.1933 [24.6 s]	dev=(HR@5:0.0469,NDCG@5:0.0277) [16.5 s]
INFO:root:Epoch 14    loss=0.1788 [25.3 s]	dev=(HR@5:0.0494,NDCG@5:0.0280) [17.8 s]
INFO:root:Epoch 15    loss=0.1648 [23.2 s]	dev=(HR@5:0.0512,NDCG@5:0.0286) [18.5 s]
INFO:root:Epoch 16    loss=0.1505 [24.2 s]	dev=(HR@5:0.0503,NDCG@5:0.0287) [20.0 s]
INFO:root:Epoch 17    loss=0.1397 [25.8 s]	dev=(HR@5:0.0525,NDCG@5:0.0306) [19.7 s] *
INFO:root:Epoch 18    loss=0.1273 [26.1 s]	dev=(HR@5:0.0494,NDCG@5:0.0292) [20.9 s]
INFO:root:Epoch 19    loss=0.1174 [22.2 s]	dev=(HR@5:0.0500,NDCG@5:0.0295) [16.6 s]
INFO:root:Epoch 20    loss=0.1079 [23.7 s]	dev=(HR@5:0.0521,NDCG@5:0.0303) [15.3 s]
INFO:root:Epoch 21    loss=0.1003 [25.0 s]	dev=(HR@5:0.0512,NDCG@5:0.0307) [20.8 s] *
INFO:root:Epoch 22    loss=0.0934 [24.6 s]	dev=(HR@5:0.0509,NDCG@5:0.0303) [24.5 s]
INFO:root:Epoch 23    loss=0.0881 [25.5 s]	dev=(HR@5:0.0531,NDCG@5:0.0316) [27.9 s] *
INFO:root:Epoch 24    loss=0.0831 [25.4 s]	dev=(HR@5:0.0555,NDCG@5:0.0326) [25.6 s] *
INFO:root:Epoch 25    loss=0.0787 [30.5 s]	dev=(HR@5:0.0543,NDCG@5:0.0319) [21.3 s]
INFO:root:Epoch 26    loss=0.0740 [32.0 s]	dev=(HR@5:0.0540,NDCG@5:0.0319) [20.5 s]
INFO:root:Epoch 27    loss=0.0700 [36.5 s]	dev=(HR@5:0.0521,NDCG@5:0.0313) [16.8 s]
INFO:root:Epoch 28    loss=0.0665 [36.2 s]	dev=(HR@5:0.0500,NDCG@5:0.0306) [19.6 s]
INFO:root:Epoch 29    loss=0.0640 [32.0 s]	dev=(HR@5:0.0531,NDCG@5:0.0319) [24.1 s]
INFO:root:Epoch 30    loss=0.0608 [26.1 s]	dev=(HR@5:0.0546,NDCG@5:0.0328) [27.7 s] *
INFO:root:Epoch 31    loss=0.0582 [26.1 s]	dev=(HR@5:0.0543,NDCG@5:0.0327) [27.1 s]
INFO:root:Epoch 32    loss=0.0563 [27.1 s]	dev=(HR@5:0.0543,NDCG@5:0.0326) [24.5 s]
INFO:root:Epoch 33    loss=0.0545 [29.3 s]	dev=(HR@5:0.0515,NDCG@5:0.0318) [23.2 s]
INFO:root:Epoch 34    loss=0.0530 [31.9 s]	dev=(HR@5:0.0537,NDCG@5:0.0331) [17.1 s] *
INFO:root:Epoch 35    loss=0.0509 [32.9 s]	dev=(HR@5:0.0549,NDCG@5:0.0336) [19.9 s] *
INFO:root:Epoch 36    loss=0.0494 [30.1 s]	dev=(HR@5:0.0546,NDCG@5:0.0336) [23.1 s] *
INFO:root:Epoch 37    loss=0.0483 [30.2 s]	dev=(HR@5:0.0540,NDCG@5:0.0338) [23.6 s] *
INFO:root:Epoch 38    loss=0.0462 [31.9 s]	dev=(HR@5:0.0546,NDCG@5:0.0339) [23.7 s] *
INFO:root:Epoch 39    loss=0.0450 [31.8 s]	dev=(HR@5:0.0558,NDCG@5:0.0342) [24.3 s] *
INFO:root:Epoch 40    loss=0.0440 [25.7 s]	dev=(HR@5:0.0509,NDCG@5:0.0319) [24.3 s]
INFO:root:Epoch 41    loss=0.0435 [29.8 s]	dev=(HR@5:0.0540,NDCG@5:0.0328) [21.3 s]
INFO:root:Epoch 42    loss=0.0424 [28.7 s]	dev=(HR@5:0.0534,NDCG@5:0.0328) [25.9 s]
INFO:root:Epoch 43    loss=0.0419 [29.6 s]	dev=(HR@5:0.0552,NDCG@5:0.0332) [23.4 s]
INFO:root:Epoch 44    loss=0.0407 [30.8 s]	dev=(HR@5:0.0558,NDCG@5:0.0334) [21.8 s]
INFO:root:Epoch 45    loss=0.0399 [25.7 s]	dev=(HR@5:0.0574,NDCG@5:0.0342) [23.4 s]
INFO:root:Epoch 46    loss=0.0398 [26.0 s]	dev=(HR@5:0.0586,NDCG@5:0.0348) [19.0 s] *
INFO:root:Epoch 47    loss=0.0389 [25.0 s]	dev=(HR@5:0.0592,NDCG@5:0.0353) [18.8 s] *
INFO:root:Epoch 48    loss=0.0390 [26.5 s]	dev=(HR@5:0.0586,NDCG@5:0.0351) [14.3 s]
INFO:root:Epoch 49    loss=0.0381 [20.9 s]	dev=(HR@5:0.0598,NDCG@5:0.0354) [16.3 s] *
INFO:root:Epoch 50    loss=0.0368 [20.9 s]	dev=(HR@5:0.0592,NDCG@5:0.0348) [16.2 s]
INFO:root:Epoch 51    loss=0.0363 [21.0 s]	dev=(HR@5:0.0549,NDCG@5:0.0335) [16.3 s]
INFO:root:Epoch 52    loss=0.0361 [20.8 s]	dev=(HR@5:0.0564,NDCG@5:0.0344) [16.5 s]
INFO:root:Epoch 53    loss=0.0360 [21.3 s]	dev=(HR@5:0.0571,NDCG@5:0.0348) [18.1 s]
INFO:root:Epoch 54    loss=0.0352 [23.4 s]	dev=(HR@5:0.0598,NDCG@5:0.0360) [15.3 s] *
INFO:root:Epoch 55    loss=0.0350 [22.2 s]	dev=(HR@5:0.0595,NDCG@5:0.0356) [14.2 s]
INFO:root:Epoch 56    loss=0.0345 [20.9 s]	dev=(HR@5:0.0595,NDCG@5:0.0362) [16.2 s] *
INFO:root:Epoch 57    loss=0.0339 [20.9 s]	dev=(HR@5:0.0604,NDCG@5:0.0362) [16.1 s]
INFO:root:Epoch 58    loss=0.0342 [21.0 s]	dev=(HR@5:0.0607,NDCG@5:0.0364) [16.3 s] *
INFO:root:Epoch 59    loss=0.0334 [20.9 s]	dev=(HR@5:0.0626,NDCG@5:0.0370) [16.2 s] *
INFO:root:Epoch 60    loss=0.0330 [20.8 s]	dev=(HR@5:0.0644,NDCG@5:0.0380) [16.2 s] *
INFO:root:Epoch 61    loss=0.0333 [20.9 s]	dev=(HR@5:0.0632,NDCG@5:0.0381) [14.8 s] *
INFO:root:Epoch 62    loss=0.0335 [22.1 s]	dev=(HR@5:0.0632,NDCG@5:0.0379) [14.6 s]
INFO:root:Epoch 63    loss=0.0324 [21.0 s]	dev=(HR@5:0.0641,NDCG@5:0.0385) [16.2 s] *
INFO:root:Epoch 64    loss=0.0320 [20.8 s]	dev=(HR@5:0.0589,NDCG@5:0.0362) [16.8 s]
INFO:root:Epoch 65    loss=0.0318 [21.1 s]	dev=(HR@5:0.0604,NDCG@5:0.0368) [16.2 s]
INFO:root:Epoch 66    loss=0.0317 [20.8 s]	dev=(HR@5:0.0629,NDCG@5:0.0378) [16.3 s]
INFO:root:Epoch 67    loss=0.0314 [20.8 s]	dev=(HR@5:0.0620,NDCG@5:0.0377) [16.3 s]
INFO:root:Epoch 68    loss=0.0309 [20.8 s]	dev=(HR@5:0.0586,NDCG@5:0.0360) [14.4 s]
INFO:root:Epoch 69    loss=0.0310 [22.1 s]	dev=(HR@5:0.0589,NDCG@5:0.0362) [15.2 s]
INFO:root:Epoch 70    loss=0.0311 [20.8 s]	dev=(HR@5:0.0592,NDCG@5:0.0360) [16.3 s]
INFO:root:Epoch 71    loss=0.0306 [21.1 s]	dev=(HR@5:0.0595,NDCG@5:0.0363) [16.2 s]
INFO:root:Epoch 72    loss=0.0307 [20.8 s]	dev=(HR@5:0.0592,NDCG@5:0.0361) [16.4 s]
INFO:root:Early stop at 72 based on dev result.
INFO:root:
Best Iter(dev)=   63	 dev=(HR@5:0.0641,NDCG@5:0.0385) [3207.7 s]
```
##### 1.2.3 NEUMF模型结果
```plaintext
INFO:root:Test Before Training: (HR@5:0.0755,NDCG@5:0.0436,HR@10:0.1588,NDCG@10:0.0704,HR@20:0.2618,NDCG@20:0.0964,HR@50:0.5441,NDCG@50:0.1518)
INFO:root:Optimizer: Adam
INFO:root:Epoch 1     loss=0.4079 [36.4 s]	dev=(HR@5:0.0374,NDCG@5:0.0206) [20.5 s] *
INFO:root:Epoch 2     loss=0.3588 [38.1 s]	dev=(HR@5:0.0282,NDCG@5:0.0179) [22.0 s]
INFO:root:Epoch 3     loss=0.3501 [40.2 s]	dev=(HR@5:0.0264,NDCG@5:0.0164) [23.4 s]
INFO:root:Epoch 4     loss=0.3423 [33.7 s]	dev=(HR@5:0.0457,NDCG@5:0.0254) [24.2 s] *
INFO:root:Epoch 5     loss=0.3308 [38.7 s]	dev=(HR@5:0.0368,NDCG@5:0.0217) [22.2 s]
INFO:root:Epoch 6     loss=0.3212 [40.0 s]	dev=(HR@5:0.0454,NDCG@5:0.0261) [22.4 s] *
INFO:root:Epoch 7     loss=0.3120 [34.4 s]	dev=(HR@5:0.0500,NDCG@5:0.0290) [23.9 s] *
INFO:root:Epoch 8     loss=0.3029 [34.1 s]	dev=(HR@5:0.0433,NDCG@5:0.0273) [25.0 s]
INFO:root:Epoch 9     loss=0.2967 [34.7 s]	dev=(HR@5:0.0555,NDCG@5:0.0326) [24.1 s] *
INFO:root:Epoch 10    loss=0.2909 [33.6 s]	dev=(HR@5:0.0647,NDCG@5:0.0375) [26.1 s] *
INFO:root:Epoch 11    loss=0.2828 [29.3 s]	dev=(HR@5:0.0558,NDCG@5:0.0333) [26.3 s]
INFO:root:Epoch 12    loss=0.2773 [32.1 s]	dev=(HR@5:0.0500,NDCG@5:0.0301) [26.2 s]
INFO:root:Epoch 13    loss=0.2735 [33.7 s]	dev=(HR@5:0.0561,NDCG@5:0.0327) [26.2 s]
INFO:root:Epoch 14    loss=0.2691 [37.0 s]	dev=(HR@5:0.0546,NDCG@5:0.0312) [24.2 s]
INFO:root:Epoch 15    loss=0.2677 [38.2 s]	dev=(HR@5:0.0543,NDCG@5:0.0314) [23.7 s]
INFO:root:Epoch 16    loss=0.2629 [34.7 s]	dev=(HR@5:0.0586,NDCG@5:0.0333) [21.7 s]
INFO:root:Epoch 17    loss=0.2588 [34.5 s]	dev=(HR@5:0.0623,NDCG@5:0.0364) [21.5 s]
INFO:root:Epoch 18    loss=0.2548 [36.5 s]	dev=(HR@5:0.0571,NDCG@5:0.0325) [22.2 s]
INFO:root:Epoch 19    loss=0.2492 [38.2 s]	dev=(HR@5:0.0601,NDCG@5:0.0349) [24.2 s]
INFO:root:Early stop at 19 based on dev result.
INFO:root:
Best Iter(dev)=   10	 dev=(HR@5:0.0647,NDCG@5:0.0375) [1128.3 s] 
```
### 2. 改进实验结果
```plaintext
#### 2.1 原BUIR_NB模型结果
===== [FINAL PERFORMANCE] =====

VALID P@10: 0.1774, R@10: 0.1681, N@10: 0.1364
VALID P@20: 0.2446, R@20: 0.2414, N@20: 0.1571
VALID P@50: 0.3439, R@50: 0.3432, N@50: 0.1851

TEST  P@10: 0.1644, R@10: 0.1559, N@10: 0.1250
TEST  P@20: 0.2256, R@20: 0.2229, N@20: 0.1440
TEST  P@50: 0.3408, R@50: 0.3401, N@50: 0.1757
```
#### 2.2 BUIR_PYG模型结果
```plaintext
===== [FINAL PERFORMANCE] =====

VALID P@10: 0.0872, R@10: 0.0813, N@10: 0.0732
VALID P@20: 0.1170, R@20: 0.1148, N@20: 0.0814
VALID P@50: 0.1894, R@50: 0.1889, N@50: 0.1016

TEST  P@10: 0.0794, R@10: 0.0738, N@10: 0.0665
TEST  P@20: 0.1085, R@20: 0.1068, N@20: 0.0757
TEST  P@50: 0.1802, R@50: 0.1798, N@50: 0.0970
```

#### 2.3 BUIR_GAT模型结果
```plaintext
===== [FINAL PERFORMANCE] =====

VALID P@10: 0.0247, R@10: 0.0226, N@10: 0.0170
VALID P@20: 0.0411, R@20: 0.0398, N@20: 0.0225
VALID P@50: 0.0878, R@50: 0.0874, N@50: 0.0352

TEST  P@10: 0.0336, R@10: 0.0315, N@10: 0.0204
TEST  P@20: 0.0535, R@20: 0.0524, N@20: 0.0271
TEST  P@50: 0.1003, R@50: 0.1001, N@50: 0.0400
```
#### 2.4 BUIR_GCN模型结果
```plaintext
===== [FINAL PERFORMANCE] =====

VALID P@10: 0.1067, R@10: 0.1044, N@10: 0.0701
VALID P@20: 0.1571, R@20: 0.1559, N@20: 0.0860
VALID P@50: 0.2460, R@50: 0.2458, N@50: 0.1077

TEST  P@10: 0.0903, R@10: 0.0884, N@10: 0.0623
TEST  P@20: 0.1264, R@20: 0.1257, N@20: 0.0739
TEST  P@50: 0.2229, R@50: 0.2228, N@50: 0.0973
```
