#set heading(numbering: "1.1")

= Task 1
== 模型架构

#figure(
  image("assets/2024-05-24-19-21-21.png",width: 80%),
  caption: [ResNet-18模型架构]
)

我们修改ResNet-18的架构, 将其输出层大小设置为200以适应数据集的类别数量.

```py
if use_pretrained:
    print("Load weights from pretrained model")
    self.resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
else:
    print("Load weights from scratch")
    self.resnet = resnet18()

self.use_pretrained = use_pretrained
self.resnet.fc = nn.Linear(512, num_classes)
```

== 数据集介绍

Caltech-UCSD Birds-200-2011 (CUB-200-2011) 数据集是用于细粒度视觉分类任务中最广泛使用的数据集。它包含11,788张鸟类的图像，共有200个子类别，其中5,994张用于训练，5,794张用于测试。每张图像都有详细的标注信息：1个子类别标签、15个部位位置、312个二值属性和1个边界框。文本信息来自Reed等人。他们通过收集细粒度的自然语言描述扩展了CUB-200-2011数据集。每张图像收集了10条单句描述。这些自然语言描述通过Amazon Mechanical Turk（AMT）平台收集，每条描述至少包含10个词，并且不包含任何关于子类别和动作的信息。

#figure(
  image("assets/2024-05-24-19-38-56.png",width:80%),
  caption:[数据集概览]
)

我们根据数据集中的`train_test_split.txt`先划分出训练集. 在剩下的每个类别的数据中, 随机以 1:1 的比例划分验证集和测试集. 

== 实验结果

#figure(
    grid(
        columns: 2,
        gutter: 2mm,
        
        
    ),
    caption: [
        Visualization of Weights of Each Neuron in Layer 1 and Layer 2
    ],
)
