# SuHou

日本立直麻将 AI 实验项目。当前阶段聚焦于牌谱处理、行为克隆基线和可重复的训练/评估流程；后续再接入 Mortal 对局与价值学习。

## 技术栈

- Python 3.12
- uv（虚拟环境与依赖管理）
- PyTorch 2.14.1（CUDA 13.0 构建）/ NumPy
- pytest / Ruff

PyTorch 通过 `pyproject.toml` 中固定的 PyTorch cu130 索引安装；若要在没有 NVIDIA GPU 的机器上使用，请改用 CPU 索引。

## 快速开始

```bash
uv sync
source .venv/Scripts/activate  # Git Bash
# PowerShell: .venv\Scripts\Activate.ps1

python -m suhou
pytest
ruff check .
```

不使用激活脚本时，也可以直接通过 uv 运行：

```bash
uv run python -m suhou
uv run pytest
uv run ruff check .
```

## 目录结构

```text
SuHou/
├── configs/      训练与数据处理配置
├── data/         牌谱与转换后的数据（不提交到 Git）
├── artifacts/    模型检查点、日志和评估结果（不提交到 Git）
├── scripts/      一次性脚本与实验入口
├── src/suhou/    Python 包
├── tests/        自动化测试
└── pyproject.toml
```

## 当前约定

- 训练/验证/测试集按完整对局划分，不能按单步决策随机划分。
- 所有模型输入只能使用当时可见的信息，不得泄漏未来摸牌或对手手牌。
- 模型输出必须通过合法动作掩码过滤。
- `data/` 与 `artifacts/` 仅保留说明文件，原始数据和大文件不入库。
