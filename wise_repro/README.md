# WISE 复现记录

复现对象是 [functions-lab/WISE](https://github.com/functions-lab/WISE) 的 commit `9e05926`。
运行环境是一台没有 GPU、没有 MATLAB 的 4 核 Linux 容器。

## 做了什么

| 部分 | 原代码 | 这里怎么跑的 |
|---|---|---|
| 复数网络训练 | `ML/main.py`，PyTorch | 原样运行，只改了命令行参数 |
| 单层模拟自测 | `Hybrid/TestFC.m`，MATLAB | 移植到 numpy 后运行，见 `run_testfc.py` |
| 数据集逐层模拟 | `Hybrid/Simulation/main_dataset.m`，MATLAB | 移植到 numpy 后运行，见 `run_dataset.py` |
| USRP 硬件实验 | `Hybrid/Dataset_new` 等 | 没有硬件，未运行 |

`wise_sim.py` 是 MATLAB 仿真链路的逐函数翻译：预处理、发射波形生成、两种信道模型、接收解码和后处理。
1 起始下标和 MATLAB 的四舍五入规则都按原样处理。与原代码不同的地方在源码里标了 `DEVIATION`。

## 复现步骤

```bash
# 1. 准备数据：从 Google 的 MNIST 镜像下载 4 个 idx.gz 文件，转换成 WISE 要求的 CSV
python prepare_mnist.py <idx 文件目录> <WISE>/ML/Data/MNIST

# 2. 训练：纯全连接，token 以 FC3 结尾，模拟脚本按这个名字找模型
cd <WISE>/ML && mkdir -p Result
python main.py --data MNIST --size 14 --type zadoff --model Hybrid --act zadoff \
  --param "h64,h32" --token FC3 --device cpu --epoch 30

# 3. 单层模拟自测与移植检查
python run_testfc.py                   # TestFC 原始参数
python run_testfc.py --power 200 --tag testfc_noiseless
python check_port.py

# 4. 数据集逐层模拟
python run_dataset.py --model <WISE>/ML/Result/MNIST_size14_FC3/model_accMax.mat \
  --data <WISE>/ML/Data/Data_MNIST_zadoff_14.mat --samples 10000
```

依赖：torch、numpy、scipy、matplotlib、tqdm、soundfile。

## 结果

### 1. 训练

在 CPU 上训练 30 轮，约 12 分钟。网络是 196、64、32、10 三层复数全连接，一次推理 14912 次乘加。

| 指标 | 值 |
|---|---|
| 最佳测试准确率 | 97.62%，第 21 轮 |
| 最后一轮测试准确率 | 97.56% |
| 最后一轮训练准确率 | 98.39% |

模型保存在 `results/model/model_accMax.mat`，训练曲线在同一目录。

### 2. 单层模拟自测

TestFC 的原始设置：100×100 随机复数矩阵，20 组，射频功率 −37 dBm，噪声系数 28 dB。

| 设置 | 信噪比 | 皮尔逊相关 | RMSE |
|---|---|---|---|
| 原始参数 | 34.8 dB | 0.9864 | 0.0825 |
| 几乎无噪声，功率 200 dBm | 272 dB | 0.9865 | 0.0822 |

带噪声和不带噪声的结果几乎一样，误差主要来自编码方式本身。`check_port.py` 定位了原因：

- **编码部分移植正确。** 两路发射波形在离散时域逐点相乘后，每个符号都在同一个频点上精确等于 x·W。
- **time 编码模式有约 2% 的固有误差。** 这个模式把 IDFT 并进权重，客户端直接发原始样本，输入频谱因此铺满整个频带。推导默认的是离散循环卷积，而连续时间里两路信号相乘得到的是线性卷积，频带边缘的几项会落到读出窗口之外。对单个符号做周期上采样后再相乘，相位误差标准差是 0.27 弧度，与完整仿真的 0.23 弧度相当。
- **freq 编码模式没有这个问题。** 剩下的误差来自滤波器泄漏，把循环前缀和保护间隔加长后，相关系数升到 0.99999。

| 编码模式 | 循环前缀 | 保护间隔 | 皮尔逊相关 | 相位误差标准差 |
|---|---|---|---|---|
| time | 默认 | 0.333 | 0.9797 | 0.237 rad |
| time | 10 | 3 | 0.9812 | 0.230 rad |
| freq | 默认 | 0.333 | 0.9991 | 0.104 rad |
| freq | 4 | 1 | 0.99992 | 0.017 rad |
| freq | 10 | 3 | 0.99999 | 0.008 rad |

### 3. 数据集逐层模拟

用训练好的模型对 MNIST 全部 10000 张测试图做模拟推理，扫描 21 个射频输入功率。
参数沿用 `main_dataset.m`：fast 信道模型、time 编码、split-4、本振功率 −3.56 dBm、插损和噪声系数为 0。
每次乘加能耗 E_MAC 按原脚本的口径计算，即射频输入功率乘以波形时长再除以乘加次数，不含本振和接收机功耗。

| 射频功率 | E_MAC | 模拟准确率 | 与数字结果一致的比例 |
|---|---|---|---|
| 数字推理 | | 97.62% | |
| −70 dBm | 2.0e-18 J | 97.17% | 99.06% |
| −76 dBm | 5.1e-19 J | 96.53% | 97.79% |
| −80 dBm | 2.0e-19 J | 93.46% | 94.52% |
| −84 dBm | 8.1e-20 J | 80.20% | 80.70% |
| −88 dBm | 3.2e-20 J | 47.04% | 47.24% |
| −92 dBm | 1.3e-20 J | 15.29% | 15.35% |
| −100 dBm | 2.0e-21 J | 10.41% | 10.41% |

功率足够时，模拟推理比数字推理低约 0.5 个百分点，这与上面 time 模式的固有误差一致。功率低于 −92 dBm 后退化为随机猜测。完整曲线见 `results/dataset_mnist_fc3.png`。

## 发现的原代码问题

- **仿真脚本选的信道模型被注释掉了。** `main_dataset.m` 设置 `transMode = "fast"`，但 `Tx2Rx_all.m` 里 fast 分支被注释，原样运行会失败。移植版把它接了回去。
- **方阵权重配上 100 的批大小会走错分支。** `LayerFC.m` 用"权重行数等于批大小且列数等于输入长度"判断逐样本内积模式。100×100 的权重遇到 100 个样本的批次就会误判，输出形状变成 100×1。作者批量训练脚本里的 h100,h100 配置正好会触发。
- **多用户的理想信道模式不可用。** `Tx2Rx_all.m` 把多行输入整块传给单用户函数，赋值也写在循环外。
- **训练脚本只有 Hybrid 和 ANN-Real 两种模型能选。** 其余三种会因为模型变量未定义报错。

## 没有做的部分

- USRP 硬件实验和信道校准，需要真实射频设备。
- 原版 MATLAB 代码本身没有运行。所有模拟结果都来自 numpy 移植版，移植的正确性由 `check_port.py` 验证。
