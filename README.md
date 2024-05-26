# Task 1

## 训练

```py
python3 ./main.py --use_pretrained 1 -epochs 50 --lr 0.001 --weight_decay 0.001
```

所有可选参数

<!-- ```py
parser = argparse.ArgumentParser(description="Train the model")
parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
parser.add_argument(
    "--weight_decay", type=float, default=0.001, help="Weight decay"
)
parser.add_argument("--momentum", type=float, default=0.9, help="Momentum")
parser.add_argument("--use_cache", type=str2bool, default=False, help="Use cache")
parser.add_argument(
    "--use_pretrained", type=str2bool, default=True, help="Use pretrained"
)
``` -->

| 参数名 | 类型 | 默认值 | 描述 |
| --- | --- | --- | --- |
| `--epochs` | int | 50 | Number of epochs |
| `--lr` | float | 0.001 | Learning rate |
| `--weight_decay` | float | 0.001 | Weight decay |
| `--momentum` | float | 0.9 | Momentum |
| `--use_cache` | bool | False | 是否使用缓存的权重 |
| `--use_pretrained` | bool | True | 是否使用预训练权重 |

## 测试

```py
from main import BirdClassificationCNN

model = BirdClassificationCNN()
model.read_model("your_weight.pth")
```

在训练集和验证集上测试

```py
model.evaluate_on_train_and_val()
```

在特定`dataloader`上测试

```py
model.evaluate(your_dataloader)
```
