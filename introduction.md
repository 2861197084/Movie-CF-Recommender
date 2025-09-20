# MovieLens 协同过滤推荐系统

基于 MovieLens 数据集的学术级协同过滤算法实现。本项目遵循软件工程最佳实践和学术研究标准，提供可重现的综合分析框架。

## 🎯 项目概述

本系统提供以下功能：

- **用户协同过滤算法**：基于相似用户偏好进行推荐
- **物品协同过滤算法**：基于物品相似性进行推荐
- **综合评估体系**：多种学术标准评估指标（MAE、RMSE、Precision@K、Recall@K、NDCG@K）
- **学术级可视化**：发表级图表和交互式仪表板
- **可重现研究**：详细日志记录、配置管理和统计分析

## 🚀 快速开始

### 环境要求

```bash
python >= 3.8
pip install -r requirements.txt
```

### 安装方法

```bash
git clone <repository-url>
cd Movie-CF-Recommender
pip install -r requirements.txt
```

### 基本使用

```bash
# 运行完整实验（自动下载数据集）
python main.py

# 快速测试（自动下载 ml-latest-small）
python main.py --quick-test

# 使用不同数据集（自动下载）
python main.py --dataset ml-25m

# 自定义实验名称
python main.py --experiment-name "我的实验"

# 禁用自动下载（需要手动准备数据）
python main.py --no-download --data-path "/path/to/movielens/data"
```

# 超参数优化
python main.py --hyperparameter-search --search-method grid_search

# 快速超参数搜索测试
python main.py --hyperparameter-search --quick-test --cv-folds 3

# 随机搜索超参数优化
python main.py --hyperparameter-search --search-method random_search --n-trials 100
```

## 🆕 超参数分析功能

系统现在支持完整的超参数优化和学术分析：

### 主要特性
- **网格搜索和随机搜索**：支持系统化的超参数空间探索
- **交叉验证评估**：k-折交叉验证确保结果稳定性
- **统计显著性测试**：Wilcoxon检验验证结果可信度
- **学术级可视化**：VMamba风格的专业图表分析
- **自动报告生成**：LaTeX和Markdown格式的学术报告

### 超参数搜索空间
- 邻居数量 (k)：[10, 20, 30, 50, 75, 100]
- 相似度指标：cosine, pearson
- 数据预处理参数：最小评分阈值
- 训练比例：[0.7, 0.8, 0.9]

### 使用示例

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

## 📁 项目结构

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

## 📊 算法实现

### 用户协同过滤算法

基于相似用户的评分模式进行推荐。

**数学基础：**
```
r̂(u,i) = r̄(u) + Σ(sim(u,v) * (r(v,i) - r̄(v))) / Σ|sim(u,v)|
```

**算法特性：**
- 多种相似度度量（余弦、皮尔逊、Jaccard）
- 均值中心化减少偏置
- 可配置邻域大小
- 处理冷启动问题

### 物品协同过滤算法

推荐与用户高评分物品相似的其他物品。

**数学基础：**
```
r̂(u,i) = r̄(i) + Σ(sim(i,j) * (r(u,j) - r̄(j))) / Σ|sim(i,j)|
```

**算法特性：**
- 物品-物品相似度计算
- 适用于用户多于物品的场景
- 提供可解释的推荐结果
- 时间稳定性好

## 📈 评估指标

### 评分预测指标
- **MAE**（平均绝对误差）
- **RMSE**（均方根误差）
- **NMAE**（归一化平均绝对误差）
- **皮尔逊相关系数**

### 排序评估指标
- **Precision@K**：推荐物品中相关物品的比例
- **Recall@K**：相关物品中被推荐物品的比例
- **F1@K**：Precision 和 Recall 的调和平均
- **NDCG@K**：归一化折损累积增益
- **Hit Rate@K**：是否命中相关物品

### 多样性指标
- **Coverage**：推荐中出现的物品覆盖率
- **Novelty**：推荐物品的新颖度

## 🔧 配置系统

系统使用全面的配置管理系统（`config.py`），包含以下部分：

- **数据配置**：数据集路径、预处理参数
- **模型配置**：相似度度量、邻域大小
- **评估配置**：计算指标、K值设置
- **可视化配置**：图表设置、输出格式
- **实验配置**：日志记录、结果存储、可重现性

### 配置示例

```python
from config import cfg

# 修改配置参数
cfg.model.user_k_neighbors = 100
cfg.evaluation.top_k_recommendations = [5, 10, 20, 50]
cfg.model.similarity_metric = "pearson"
```

## 📊 可视化系统

系统生成发表级别的可视化结果：

### 静态图表
- 数据集统计和分布分析
- 模型性能对比
- 相似度矩阵热力图
- 评分分布分析
- 性能指标趋势图

### 交互式仪表板
- 基于 Plotly 的交互式分析
- 模型对比工具
- 深入分析功能
- 结果导出功能

## 🔬 学术特性

### 研究标准
- 时间戳完整日志记录
- 可重现随机种子
- 统计显著性检验
- 详细实验文档

### 软件工程
- 模块化可扩展架构
- 类型提示和文档
- 错误处理和验证
- 单元测试兼容性

### 输出格式
- JSON 结果便于程序访问
- CSV 摘要便于表格分析
- 学术图表（PNG、PDF）
- 交互式 HTML 仪表板

## 📥 数据要求

### 自动下载（推荐）

系统自动下载和处理 MovieLens 数据集，无需手动设置！

只需运行：
```bash
python main.py --quick-test  # 自动下载 ml-latest-small
```

### 手动设置（可选）

如果您偏好手动设置，请从 [GroupLens](https://grouplens.org/datasets/movielens/) 下载：

1. 下载 MovieLens 数据集（ml-latest-small 或 ml-latest）
2. 解压到 `./data/` 目录
3. 确保 `ratings.csv` 和 `movies.csv` 文件存在
4. 运行时使用 `--no-download` 标志

### 数据格式

**ratings.csv:**
```
userId,movieId,rating,timestamp
1,1,4.0,964982703
1,3,4.0,964981247
```

**movies.csv:**
```
movieId,title,genres
1,Toy Story (1995),Adventure|Animation|Children|Comedy|Fantasy
2,Jumanji (1995),Adventure|Children|Fantasy
```

## 🏃‍♂️ 高级用法

### 自定义评估

```python
from evaluation.metrics import MetricsEvaluator
from models.user_based_cf import UserBasedCollaborativeFiltering

# 初始化模型和评估器
model = UserBasedCollaborativeFiltering(similarity_metric="cosine", k_neighbors=50)
evaluator = MetricsEvaluator()

# 训练模型
model.fit(user_item_matrix)

# 综合评估
results = evaluator.comprehensive_evaluation(
    model=model,
    test_data=test_df,
    user_item_matrix=user_item_matrix,
    k_values=[5, 10, 20],
    threshold=3.5
)
```

### 自定义相似度度量

```python
# 扩展基类实现自定义相似度
class CustomCF(BaseCollaborativeFiltering):
    def compute_similarity_matrix(self, data_matrix):
        # 实现自定义相似度计算
        pass
```

### 批量预测

```python
# 高效批量预测
user_item_pairs = [(1, 100), (1, 200), (2, 100)]
predictions = model.predict_batch(user_item_pairs)
```

## 📚 学术参考

本实现遵循已建立的研究方法论：

1. **协同过滤基础**
   - Resnick, P., et al. (1994). GroupLens: An open architecture for collaborative filtering of netnews
   - Herlocker, J. L., et al. (1999). An algorithmic framework for performing collaborative filtering

2. **基于物品的方法**
   - Sarwar, B., et al. (2001). Item-based collaborative filtering recommendation algorithms
   - Deshpande, M., & Karypis, G. (2004). Item-based top-n recommendation algorithms

3. **评估方法论**
   - Herlocker, J. L., et al. (2004). Evaluating collaborative filtering recommender systems
   - Shani, G., & Gunawardana, A. (2011). Evaluating recommendation systems

## 🤝 贡献指南

本项目遵循学术软件开发标准：

1. **代码质量**：PEP 8 规范、类型提示、文档注释
2. **测试**：所有组件的单元测试
3. **文档**：遵循学术标准的文档字符串
4. **可重现性**：固定随机种子、全面日志记录





## 📞 使用说明

### 最简单的开始方式

```bash
# 克隆项目
git clone <repository-url>
cd Movie-CF-Recommender

# 安装依赖
pip install -r requirements.txt

# 开始实验（自动下载数据）
python main.py --quick-test
```

### 预期输出

运行后将生成：
- `./results/`: 实验结果（JSON、CSV格式）
- `./plots/`: 可视化图表（PNG、PDF格式）
- `./logs/`: 详细日志文件

### 故障排除

如果遇到问题：
1. 检查网络连接（用于自动下载）
2. 尝试使用 `--dataset ml-100k`（较小数据集）
3. 使用 `--no-download` 并手动下载数据
4. 查看日志文件获取详细错误信息