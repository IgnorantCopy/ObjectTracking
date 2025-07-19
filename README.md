# ObjectTracking
A repository for object tracking/detection for low-altitude radar.

## 1. 单帧 RD 图分类方法

> 对应 frame_wise 目录

1. 实现模型：ViT、SwinTransformer
2. 待完善：./frame_wise/models/cnn.py 以及 ./frame_wise/visualize.py
    + 前者近期不打算实现
    + 后者打算用 Grad-CAM 算法来可视化特征图的重要性，待实现及验证
3. DataLoader：将 RD 图独立加载，无**序列**的概念，可通过运行 ./data/RDMap_new.m 文件获取 RD 图

## 2. 多帧分类方法

> 对应 seq_wsie 目录

1. 实现模型：ViT in ViT、ViViT、SwinTransformer3D
2. DataLoader：将一条航迹的 RD 图视为一个整体，包括计算准确率时
    + 注意这与赛方要求不一致

> 注意：该版本开始 RD 图直接从原始回波中提取，不再提前准备并保存

> flash attention Windows 版本下载地址：https://github.com/kingbri1/flash-attention/releases
> 
> Linux 版本下载地址：https://github.com/Dao-AILab/flash-attention/releases

## 3. RD 图与点航数据融合方法

> 对应 fusion 目录

1. 实现模型：将 seq_wise 中的 SwinTransformer3D 与点航数据的 RoFormer 两个模型进行结合
    + 目前直接将两者分类头之前的输出进行拼接，如果有更好的方案，欢迎 PR
2. 新增功能：运行 train.py 时，可通过指定 --result-path 参数来保存最佳模型在数据集上的分类结果，以可视化的方式保存，结构如下：

```shell
└── result_path
    ├── train
    │   ├── correct
    │   │   ├── Label_1
    │   │   │   ├── Batch_1333
    │   │   │   │   ├── Frame_1
    │   │   │   │   │   ├── rd_map.png
    │   │   │   │   │   └── trajectory.png
    │   │   │   │   ├── Frame_2
    │   │   │   │   └── ...
    │   │   │   └── ...
    │   │   ├── Label_2
    │   │   └── ...
    │   └── wrong
    │       ├── Label_1
    │       ├── Label_2
    │       └── ...
    └── val
        ├── correct
        │   └── ...
        └── wrong
            └── ...
```

> 注：rd_map.png 上标注了模型预测的标签，文件夹名称是真实标签

3. 待完善： 
   + RD 图分支考虑将全局信息融合进来，可从位置编码着手

## 4. 训练杂波/噪声二分类预训练模型
> 对应 noise_cls 目录

1. 数据准备：
   + 通过运行 data/RDMap_new.m 文件得到原始 RD 图数据，得到 MTD_Results 目录
   + 运行 noise_cls/utils/data.py，读取 MTD_Results 目录，生成本任务的数据集 noise_cls
2. 模型架构：简单的三层全连接层

## 5. 提示词工程
> 对应 prompt_engineering 目录

> 仅实现了代码，效果未验证（缺钱😜）
