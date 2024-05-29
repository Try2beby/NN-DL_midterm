#set heading(numbering: "1.1")

#set quote(block: true)

#quote[
    repo 地址: https://github.com/He1senbergg/MidTerm
]

= Task 1
== 模型架构

#figure(
  image("assets/2024-05-24-19-21-21.png",width: 80%),
  caption: [ResNet-18模型架构]
)

我们修改ResNet-18的架构, 将其输出层大小设置为200以适应数据集的类别数量.

```py
model = models.resnet18()
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)
```

== 数据集介绍

Caltech-UCSD Birds-200-2011 (CUB-200-2011) 数据集是用于细粒度视觉分类任务中最广泛使用的数据集。它包含11,788张鸟类的图像，共有200个子类别，其中5,994张用于训练，5,794张用于测试。每张图像都有详细的标注信息：1个子类别标签、15个部位位置、312个二值属性和1个边界框。文本信息来自Reed等人。他们通过收集细粒度的自然语言描述扩展了CUB-200-2011数据集。每张图像收集了10条单句描述。这些自然语言描述通过Amazon Mechanical Turk（AMT）平台收集，每条描述至少包含10个词，并且不包含任何关于子类别和动作的信息。

#figure(
  image("assets/2024-05-24-19-38-56.png",width:80%),
  caption:[数据集概览]
)

我们根据数据集中的`train_test_split.txt`先划分出训练集. 在剩下的每个类别的数据中, 随机以 1:1 的比例划分验证集和验证集. 

== 实验结果

=== 预训练权重初始化（除fc层外）

使用预训练权重时, 置全连接层学习率为 `lr`, 全连接层以外的学习率 `0.1 * lr`, 具体设置方式为
```py
parameters = [
{"params": model.fc.parameters(), "lr": learning_rate},
{"params": [param for name, param in model.named_parameters() if "fc" not in name], "lr": learning_rate*0.1}
]
```
另外, 我们还添加了学习率递减策略, 在第 `num_epochs * 0.5` 和 `num_epochs * 0.75` 将学习率乘以 `gamma` (`gamma < 1`).

我们尝试了两种优化器: SGD 和 Adam.

对于SGD, 考虑到在验证集上准确率较低, 可能存在过拟合, 我们调整测试了三个 `weight_decay` 值, 分别为 `0.001, 0.01, 0.05`. 结果表明, `weight_decay = 0.01` 时在验证集上准确率较高 (0.682 > 0.677 > 0.675), 损失函数值较低, 但三者并无明显差距. 因此在图像中呈现的效果也是三者相近.

// #table(
//   align: center+horizon,
//     columns: 2,
//     [`weight_decay`],[`lr`],
//     [0.001],[0.001],
//     [0.01],[0.001],
//     [0.05],[0.001]
// )

#figure(
    grid(
        columns: 2,
        gutter: 2mm,
        image("./assets/Accuracy_Train_Accuracy.svg"),
        image("./assets/Accuracy_Test_Accuracy.svg")
        
    ),
    caption: [
        训练集和验证集上 accuracy 变化 (SGD)
    ],
)

#figure(
    grid(
        columns: 2,
        gutter: 2mm,
        image("./assets/Loss_Train_Loss.svg"),
        image("./assets/Loss_Test_Loss.svg")
        
    ),
    caption: [
        训练集和验证集上 loss 变化 (SGD)
    ],
)

对于Adam, 我们同样测试了上述三个 `weight_decay` 值. 结果表明 `weight_decay` 值对最终准确率和损失函数值无明显影响, 其主要影响二者图像中波动的起止点. 另一个明显的问题是相较于SGD, Adam迭代过程中出现了明显的波动现象, 尤其是在验证集上, 可能原因是学习率设置过大, 或者 `batch_size` 过小引入了高方差.

#figure(
    grid(
        columns: 2,
        gutter: 2mm,
        image("./assets/adam_Accuracy_Train_Accuracy.svg"),
        image("./assets/adam_Accuracy_Test_Accuracy.svg")
        
    ),
    caption: [
        训练集和验证集上 accuracy 变化 (Adam)
    ],
)

#figure(
    grid(
        columns: 2,
        gutter: 2mm,
        image("./assets/adam_Loss_Train_Loss.svg"),
        image("./assets/adam_Loss_Test_Loss.svg")
        
    ),
    caption: [
        训练集和验证集上 loss 变化 (Adam)
    ],
)

=== 随机初始化

我们还测试了随机初始化网络参数的效果 (`weight_decay = 0.01, lr = 0.001`). 与使用预训练权重相比, 虽然也能在训练集上达到较高的准确率, 但在验证集上准确率和损失函数值的变化明显不稳定, 且最终验证集上准确率 (0.17) 明显低于使用预训练权重时的情形, 表明模型泛化能力明显减弱.

#figure(
    grid(
        columns: 2,
        gutter: 2mm,
        image("./assets/s_Accuracy_Train_Accuracy.svg"),
        image("./assets/s_Accuracy_Test_Accuracy.svg")
        
    ),
    caption: [
        训练集和验证集上 accuracy 变化 (SGD)
    ],
)

#figure(
    grid(
        columns: 2,
        gutter: 2mm,
        image("./assets/s_Loss_Train_Loss.svg"),
        image("./assets/s_Loss_Test_Loss.svg")
        
    ),
    caption: [
        训练集和验证集上 loss 变化 (SGD)
    ],
)