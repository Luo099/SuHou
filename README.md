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

重放状态中的舍牌和副露额外保留内部字段：舍牌带 `tile_id`/`seq`，副露带 `formed_seq`/`updated_seq`/`added_tile`。加杠的 `called_tile` 是原来碰入的牌，`added_tile` 是加杠追加的牌，二者不再混为同一个字段。这些字段用于观察构造和牌张流转检查，不进入 observation。

本地 `tests/` 目录已加入 `.gitignore`，仅用于开发验证，不纳入版本控制。

## Observation v1

`src/suhou/observation/` 提供结构化的玩家可见局面。observation 的构造器只有一个入口 `build_observation`；逐决策提取可使用包装它的 `iter_decision_points`：

```python
from suhou.observation import RulesSnapshot, iter_decision_points

def rules_state(round_state, event, viewer):
    # 规则引擎接入前的占位。furiten 必须由规则引擎计算；
    # RulesSnapshot 只是显式的最小适配器，不是规则实现。
    return RulesSnapshot((False, False, False, False))

points = list(iter_decision_points(
    "data/raw/tenhou/2023041911gm-0001-0000-cba4cded.mjlog",
    rules_state,
))
observation = points[0].observation
print(observation.to_dict())
```

- observation 包含 `viewer`、`table`、`self`、`players`、`context`、`history`。`players` 始终按绝对座位 0..3 排列；`self` 只给出当前观察者的手牌、摸入牌和振听。
- 牌张使用 `Tile(kind, red)` 表示，赤五保留为 `5m/5p/5s + red=True`，不包含实体牌编号。手牌按牌种、赤牌标记排序（同牌种赤牌在前）；舍牌、副露和历史按发生顺序排列。
- `build_observation(replay_state, rules_state, visible_history, viewer, decision_context)` 逐字段读取当前重放状态，返回不可变快照。它不读取对手手牌内容、对手摸入牌、里宝牌或结算内部字段。
- `history` 是本局开始到本次决策前的可见事件前缀。自己的起手牌和摸牌有实际牌张，对手摸牌只有 `tile=None`；对手的起手牌不会出现在历史里。后续鸣牌只追加新事件，不回写旧样本。
- `seq` 是局内源事件序号，每局从 `INIT` 的 0 重新开始；舍牌和副露的 `seq` 与历史事件一致。`trigger_seq` 是触发本次决策的摸牌/舍牌/鸣牌事件。
- `table.match_type` 从天凤 `<GO type>` 规则位解出（`0x08` 为半庄，`0x10` 为三麻）。当前样例 `type=1` 是东风战；它在东4局无人达到 30000 点后南入到南1局，因此 `match_type` 为 `tonpu`，不是半庄。
- `DecisionPhase` 目前由提取层按源事件给出（`discard`/`response`/`riichi`/`self_kan`/`tsumo`），属于规则引擎接入前的临时适配。合法动作集合、动作掩码和实际动作标签不放进 observation，仍留给 sample 层。
- `RulesState` 目前只要求 `furiten(viewer)`。真实振听、临时振听和立直振听必须由复用的规则引擎提供；`RulesSnapshot` 没有默认值，上面显式传入的四家 `False` 只是占位，不能当成规则结果。
- `iter_decision_points` 是逐实际动作截取 observation 的便捷入口，内部通过 `replay_tenhou(..., before_event=...)` 在事件执行前取数。它不会为未记录的“跳过鸣牌/跳过荣和”补样本；立直宣言与随后的舍牌是两个源事件，会各自产生一个 decision point，sample 层可按需合并。
- `DecisionPoint` 额外带 `round_ordinal`（局实例序号，连庄时同一 `seed[0]` 仍可区分）、`event_seq`（历史截止位置）和源事件标识。这些是审计/落盘字段，不属于 observation 契约。

## 当前约定

- 训练/验证/测试集按完整对局划分，不能按单步决策随机划分。
- 所有模型输入只能使用当时可见的信息，不得泄漏未来摸牌或对手手牌。
- 模型输出必须通过合法动作掩码过滤。
- `data/` 与 `artifacts/` 仅保留说明文件，原始数据和大文件不入库。
