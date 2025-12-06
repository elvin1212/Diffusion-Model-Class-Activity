# Diffusion Model Implementation

这是一个基于PyTorch的简单扩散模型实现，使用CIFAR-10数据集进行训练。

## 目录结构

- `model.py`: 定义扩散模型架构
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
1. 下载并准备CIFAR-10数据集（已配置国内镜像源加速）
2. 初始化扩散模型
3. 训练模型
4. 生成样本图像
5. 保存训练好的模型

## 配置

可以在 `main.py` 中修改以下参数：
- `batch_size`: 批处理大小
- `epochs`: 训练轮数
- `learning_rate`: 学习率
- `num_samples`: 生成样本数量
- `diffusion_steps`: 生成过程中反向扩散步骤数

## 工作原理

1. **模型**: 使用简单的CNN架构预测噪声
2. **训练**: 通过添加噪声到真实图像并训练模型预测这些噪声
3. **生成**: 从纯噪声开始，逐步去除噪声生成新图像

## 数据集下载加速

为了加速CIFAR-10数据集在中国的下载速度，我们已经在代码中配置了多个国内镜像源：
- 清华大学镜像源
- 南京大学镜像源
- 中国科学技术大学镜像源
- 德国NTMM镜像源

如果需要手动下载数据集，可以尝试使用以下命令：
```bash
# 使用wget和清华镜像源
wget https://mirrors.tuna.tsinghua.edu.cn/pytorch/cifar-10-python.tar.gz
```