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

# 超参数优化
python main.py --hyperparameter-search --search-method grid_search

# 快速超参数搜索测试
python main.py --hyperparameter-search --quick-test --cv-folds 3

# 随机搜索超参数优化
python main.py --hyperparameter-search --search-method random_search --n-trials 100
```

```bash
# 完整超参数优化
python main.py --hyperparameter-search

# 自定义搜索参数
python main.py --hyperparameter-search \
               --search-method random_search \
               --n-trials 50 \
               --cv-folds 5
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

## 实验输出说明

每次运行脚本时，系统会依据时间戳、实验模式（baseline/hyperparameter）、数据集与 `--experiment-name` 自动生成独立的实验目录。例如：

```
results/20250919-205749_baseline_ml-latest-small_movielens-cf-baseline_quick-true/
plots/20250919-205749_baseline_ml-latest-small_movielens-cf-baseline_quick-true/
logs/20250919-205749_baseline_ml-latest-small_movielens-cf-baseline_quick-true/
```

这样可以避免多次实验互相覆盖。每个目录包含：

- `results.json`：完整配置、指标与数据统计
- `summary.csv`：各模型关键指标汇总
- `interactive_dashboard.html`：交互式可视化仪表盘
- `*.png`/`*.pdf`：分离保存的单张图表（数据集概览、模型表现、排名指标等）
- `academic_report_*.tex/.md`：自动生成的学术报告

## 多目标与超参数搜索

如果需要多目标调参，可在 `config.py` 中调整：

```
cfg.hyperparameter.secondary_objectives = [OptimizationObjective.PRECISION_AT_K]
cfg.hyperparameter.objective_weights = {"rmse": 0.7, "precision_at_k": 1.3}
```

运行时可通过 `--hyperparameter-search` 与 `--search-method` 指定搜索方式。所有输出会归档到对应的 run 目录，便于后续比对与复现。
