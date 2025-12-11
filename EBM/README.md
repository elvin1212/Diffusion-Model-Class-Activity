# Energy-Based Model (EBM) Implementation

这是一个基于PyTorch的简单Energy-Based模型实现，使用CIFAR-10数据集进行训练。

## 目录结构

- `model.py`: 定义EBM模型架构
- `trainer.py`: 包含训练相关的函数
- `generator.py`: 包含生成样本的函数
- `dataset.py`: 处理CIFAR-10数据集
- `main.py`: 主执行文件
- `requirements.txt`: 依赖列表

## 安装依赖

```bash
pip install -r requirements.txt
```

## 运行

```bash
python main.py
```

这将：
1. 下载并准备CIFAR-10数据集（存储在 ../data 目录下）
2. 初始化EBM模型
3. 训练模型
4. 生成样本图像
5. 保存训练好的模型

## 配置

可以在 `main.py` 中修改以下参数：
- `batch_size`: 批处理大小
- `epochs`: 训练轮数
- `learning_rate`: 学习率
- `num_samples`: 生成样本数量
- `steps`: Langevin采样步数
- `step_size`: 梯度步长

## 工作原理

1. **模型**: 使用简单的CNN架构计算输入图像的能量值
2. **训练**: 通过对比散度(Contrastive Divergence)方法训练模型，使得真实图像的能量较低，而合成图像的能量较高
3. **生成**: 使用Langevin Sampling从随机噪声开始生成新的图像样本

## 数据集

项目使用CIFAR-10数据集，并将其存储在 `../data` 目录下。为了加速在中国的下载速度，已经配置了多个国内镜像源：
- 清华大学镜像源
- 南京大学镜像源
- 中国科学技术大学镜像源
- 德国NTMM镜像源