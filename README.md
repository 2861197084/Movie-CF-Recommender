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

# 贝叶斯优化超参数搜索
python main.py --hyperparameter-search --search-method bayesian_optimization --n-trials 60

# 启用 PyTorch 后端并使用 GPU
python main.py --backend torch --device cuda --hyperparameter-search --search-method random_search --n-trials 20
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

### 数据划分策略（--split-strategy）

系统支持三种学术常用的数据划分方式：

- random：全局随机划分（默认）。
- temporal_user：按用户时间序列保留最近 test_ratio 比例为测试集（每个用户均保持因果性）。
- temporal_global：按全局时间点分割，阈值取 rating 时间的 train_ratio 分位（训练集在阈值之前，测试集在之后）。

示例：

```bash
# 每用户时间划分（更贴近线上推荐的因果评估）
python main.py --quick-test --split-strategy temporal_user

# 全局时间点划分（更严格的时序外推场景）
python main.py --quick-test --split-strategy temporal_global
```

提示：时间划分下，训练仅使用 train 子集构建矩阵与映射（避免数据泄漏）；测试集中未在训练出现的用户/物品会被跳过并在日志中汇总统计。

### 时间感知 CF 与相似度正则化参数

在 `config.py` 的 `cfg.model` 下可用的关键参数：

- temporal_decay_half_life：时间衰减半衰期（天），建议尝试 [14, 30, 60, 90]
- temporal_decay_floor：时间权重下界，建议尝试 [0.0, 0.05, 0.1]
- temporal_decay_on_similarity：是否在构建相似度前做全局时间加权（True/False）
- similarity_shrinkage_lambda：显著性收缩 λ（10~50 常见），抑制低共评样本带来的噪声
- truncate_negative_similarity：是否截断负相似度（True/False），与 Pearson 常搭配

常见组合建议：

- ItemCF：cosine + k∈[50,120]，适度 shrinkage（≈25）
- UserCF：cosine/pearson + k∈[30,100]
- Temporal：half_life 与 decay_on_similarity 两套路径都建议分别试验

### 更多命令示例

```bash
# 全局时间划分 + 随机搜索（含 temporal 与非 temporal 模型）
python main.py \
  --backend numpy \
  --split-strategy temporal_global \
  --hyperparameter-search \
  --search-method random_search \
  --n-trials 40 \
  --cv-folds 3 \
  --experiment-name temporal-global-search

# 贝叶斯优化 + 每用户时间划分
python main.py \
  --backend numpy \
  --split-strategy temporal_user \
  --hyperparameter-search \
  --search-method bayesian_optimization \
  --n-trials 30 \
  --cv-folds 3
```

### 支持的数据集

系统可自动下载和处理多种 MovieLens 数据集：

- `ml-latest-small`：约10万评分（默认，推荐测试使用）
- `ml-latest`：约2700万评分（完整数据集）
- `ml-25m`：2500万评分（稳定版本）
- `ml-100k`：10万评分（经典格式）

## 平台特定完整实验流程

### 🍎 Mac (Apple Silicon) 完整实验流程

Apple Silicon (M1/M2/M3/M4) 芯片可以利用 Metal Performance Shaders (MPS) 加速：

```bash
# 1. 检查系统信息和可用后端
python main.py --show-system-info

# 2. 快速验证 MPS 加速是否工作
python main.py --backend torch --device mps --quick-test

# 3. 性能基准测试（对比不同后端）
python test_mac_backends.py

# 4. 中等规模完整实验（推荐先运行，5-10分钟）
python main.py \
  --dataset ml-latest-small \
  --backend torch \
  --device mps \
  --hyperparameter-search \
  --search-method grid_search \
  --cv-folds 5 \
  --experiment-name "mac-mps-experiment"

# 5. 大规模生产级实验（30-60分钟）
python main.py \
  --dataset ml-25m \
  --backend torch \
  --device mps \
  --hyperparameter-search \
  --search-method random_search \
  --n-trials 200 \
  --cv-folds 5 \
  --experiment-name "mac-production"

# 自动选择最优配置（推荐）
python main.py --backend auto --device auto --hyperparameter-search
```

**Mac Intel 芯片**：使用 `--backend numpy` 或 `--backend torch --device cpu`

### 💻 Windows 完整实验流程

Windows 系统可以利用 NVIDIA GPU (CUDA) 加速：

```bash
# 1. 检查系统信息和 CUDA 可用性
python main.py --show-system-info

# 2. 安装 PyTorch with CUDA (如果尚未安装)
# 访问 https://pytorch.org/get-started/locally/ 获取适合您 CUDA 版本的命令
# 例如 CUDA 12.1:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3. 快速验证 GPU 加速
python main.py --backend torch --device cuda --quick-test

# 4. 性能对比测试
python test_mac_backends.py --all

# 5. 中等规模完整实验（5-10分钟）
python main.py ^
  --dataset ml-latest-small ^
  --backend torch ^
  --device cuda ^
  --hyperparameter-search ^
  --search-method grid_search ^
  --cv-folds 5 ^
  --experiment-name "windows-gpu-experiment"

# 6. 大规模生产级实验（20-40分钟 with GPU）
python main.py ^
  --dataset ml-25m ^
  --backend torch ^
  --device cuda ^
  --hyperparameter-search ^
  --search-method random_search ^
  --n-trials 200 ^
  --cv-folds 5 ^
  --experiment-name "windows-production"

# 无 GPU 时使用 CPU
python main.py --backend numpy --hyperparameter-search

# 自动检测并选择最优配置
python main.py --backend auto --device auto --hyperparameter-search
```

**注意**：Windows 命令行使用 `^` 作为续行符（而不是 `\`）

### 🐧 Linux 完整实验流程

Linux 系统通常有最好的 CUDA 支持：

```bash
# 1. 检查系统信息
python main.py --show-system-info

# 2. 验证 GPU (如果有 NVIDIA GPU)
nvidia-smi  # 查看 GPU 信息
python main.py --backend torch --device cuda --quick-test

# 3-6. 实验流程与 Windows 相同，但使用 \ 作为续行符
python main.py \
  --dataset ml-25m \
  --backend torch \
  --device cuda \
  --hyperparameter-search \
  --search-method random_search \
  --n-trials 200 \
  --experiment-name "linux-production"
```

### 后端性能对比

| 平台 | 后端配置 | 相对性能 | 推荐场景 |
|------|---------|---------|---------|
| Mac M1-M4 | torch + mps | 5-10x | 推荐，充分利用 Apple Silicon |
| Mac Intel | numpy | 1x (基准) | 默认选择 |
| Windows + NVIDIA | torch + cuda | 10-50x | 强烈推荐，最佳性能 |
| Windows 无GPU | numpy | 1x | 默认选择 |
| Linux + NVIDIA | torch + cuda | 10-50x | 最佳性能 |

### 自动后端选择

使用 `--backend auto --device auto` 可以自动检测并选择最优配置：
- 有 NVIDIA GPU → 使用 CUDA
- Mac Apple Silicon → 使用 MPS
- 其他情况 → 使用 CPU

### 后端与算法支持

默认的 NumPy 后端支持所有功能。PyTorch 后端支持：
- ✅ 余弦相似度 (cosine)
- ✅ Pearson 相关系数 (pearson)
- ✅ Jaccard 相似度 (jaccard)
- ✅ 稀疏矩阵优化
- ✅ 批量预测向量化
- ✅ GPU/MPS 加速

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

- `results.json` 存放所有配置、指标和统计数据，现在包含 `ranking_summary`（平均倒数排名 MRR）以及可用时的 `diversity_novelty`（覆盖率、推荐新颖度）字段
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
