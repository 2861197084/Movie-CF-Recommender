# MovieLens 协同过滤推荐系统

### 环境要求

```bash
python >= 3.8
pip install -r requirements.txt
```

### 基本使用

```bash
# 运行完整实验
python main.py

# 快速测试（自动下载 ml-latest-small）
python main.py --quick-test

# 使用不同数据集
python main.py --dataset ml-25m

# 自定义实验名称
python main.py --experiment-name "我的实验"

# 禁用自动下载
python main.py --no-download --data-path "/path/to/movielens/data"
```

### 支持的数据集

系统可自动下载和处理多种 MovieLens 数据集：

- `ml-latest-small`：约10万评分（默认，推荐测试使用）
- `ml-latest`：约2700万评分（完整数据集）
- `ml-25m`：2500万评分（稳定版本）
- `ml-100k`：10万评分（经典格式）

## 项目结构

```
Movie-CF-Recommender/
├── config.py                 # 配置管理
├── main.py                   # 主执行脚本
├── requirements.txt          # Python 依赖包
├── utils/
│   ├── logger.py             # 学术级日志系统
│   ├── data_loader.py        # MovieLens 数据处理
│   └── dataset_downloader.py # 自动数据集下载
├── models/
│   ├── base_cf.py            # 协同过滤基类
│   ├── user_based_cf.py      # 用户协同过滤实现
│   └── item_based_cf.py      # 物品协同过滤实现
├── evaluation/
│   └── metrics.py            # 综合评估指标
├── analysis/
│   └── visualizer.py         # 学术可视化套件
└── data/                     # 数据集目录（自动创建）
    ├── ratings.csv           # MovieLens 评分数据
    └── movies.csv            # MovieLens 电影数据
```
