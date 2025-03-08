# 旅行时间预测项目

这个项目使用机器学习方法（主要是随机森林回归器）来预测交通旅行时间。该系统整合了历史交通数据、天气条件、道路拥堵指标和假日信息等多种数据源，以提供准确的旅行时间预测。

## 项目结构

```
travel_time_prediction/
│
├── data/                    # 数据文件夹
│   ├── raw/                 # 原始数据
│   └── processed/           # 处理后的数据
│
├── src/                     # 源代码
│   ├── data/                # 数据处理脚本
│   ├── features/            # 特征工程脚本
│   ├── models/              # 模型训练和预测脚本
│   └── utils/               # 工具函数
│
├── notebooks/               # Jupyter notebooks用于探索性分析
│
├── requirements.txt         # 项目依赖
└── README.md                # 项目说明
```

## 安装

1. 克隆此仓库
2. 创建虚拟环境并安装依赖

```bash
python -m venv venv
source venv/bin/activate  # 在Windows上使用 venv\Scripts\activate
pip install -r requirements.txt
```

## 数据准备

1. 将原始数据放在`data/raw/`目录下
2. 运行数据处理脚本生成处理后的数据集

```bash
python src/data/make_dataset.py
```

## 特征工程

运行特征工程脚本来创建和选择模型所需的特征：

```bash
python src/features/build_features.py
```

## 模型训练

训练随机森林回归模型：

```bash
python src/models/train_model.py
```

## 模型评估

评估模型性能：

```bash
python src/models/evaluate_model.py
```

## 预测

使用训练好的模型进行预测：

```bash
python src/models/predict_model.py
```

## 项目流程

1. **数据收集与整合**：整合历史交通记录、天气数据、道路拥堵指标和假日安排等。
2. **探索性数据分析与预处理**：清洗数据，处理缺失值和异常值，编码分类变量。
3. **特征工程**：创建有意义的变量，如高峰时段指标、道路段密度和节假日标记等。
4. **特征选择**：使用递归特征消除和互信息分数等方法选择最有影响力的预测因子。
5. **模型训练与验证**：划分数据集，训练随机森林回归器，优化超参数。
6. **迭代优化与评估**：使用交叉验证确保模型可靠性，根据误差分析调整特征工程过程。
7. **部署与持续监控**：将验证后的模型部署到生产环境，建立监控机制追踪模型性能。 