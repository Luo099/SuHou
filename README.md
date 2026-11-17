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

## 天凤牌谱重放

支持天凤 mjlog 2.3 的 XML 文件（可按 gzip 压缩），保留原始 XML 子节点顺序并逐事件重放完整牌局：

```bash
uv run python -m suhou tenhou-replay \
  data/raw/tenhou/2023041911gm-0001-0000-cba4cded.mjlog \
  --output artifacts/tenhou/2023041911gm-0001-0000-cba4cded
```

输出目录包含：

- `replay.jsonl`：每个原始事件一行，含事件序号、原标签/属性、规范化动作和执行后的完整牌局状态；和了记录展开符数、飜数、点数、役种、和了牌及宝牌。
- `replay.log`：与 JSONL 相同事件顺序的中文可读动作日志，和了行可直接对照官方结算页。
- `summary.json`：玩家、局数、最终持有分数和场分。

`--step N` 仅将终端打印限制到第 N 个事件（从 0 开始）；文件始终覆盖整场牌局。程序验证实体牌副本、行动者、副露来源、立直两阶段、和了手牌/副露/宝牌、各局分数连续性及最终 `owari`，并支持四人牌谱中的吃、碰、明杠、暗杠、加杠、和了和流局记录。状态快照同时给出每家舍牌、手牌、副露、立直阶段、牌墙余数和供托。

风位在快照中以结构化字段给出：`round.round_wind` 是场风枚举（`东/南/西/北`，由 `seed[0] // 4` 解出），`round.players[].seat_wind` 是自风枚举（按 `(seat - dealer + 4) % 4` 相对庄家解出，庄家为东）。两者为 JSON 字符串，`round.name`（如「东1局」）仍保留便于对照。

终局为流局且供托仍有剩余时，按天凤规则将剩余立直棒结算给当时的第一名（并列时按起家顺序），每根 1000 点。该笔只出现在 `owari`、不包含在终局的 `sc` 增减分中；重放会在校验前补上这笔，并在 `replay.jsonl` 的 `terminal_riichi_sticks` 字段中记录归属。

内部点数统一为“点”；XML 的 `ten`、`sc` 是 100 点单位，`owari` 的场分是独立的小数值。程序忠实重放牌谱并展示记录中的役种，不自行计算役种/和了点数。天凤官方读谱仅用于验收时的正确性对照，不属于程序交付功能或运行依赖。

本地 `tests/` 目录已加入 `.gitignore`，仅用于开发验证，不纳入版本控制。

## 当前约定

- 训练/验证/测试集按完整对局划分，不能按单步决策随机划分。
- 所有模型输入只能使用当时可见的信息，不得泄漏未来摸牌或对手手牌。
- 模型输出必须通过合法动作掩码过滤。
- `data/` 与 `artifacts/` 仅保留说明文件，原始数据和大文件不入库。
