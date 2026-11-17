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
├── src/suhou/
│   ├── domain/   通用麻将类型、统一事件、状态视图与可见性投影
│   ├── replay/   通用重放引擎与 events.jsonl 读写
│   ├── tenhou/   天凤 XML 解析、协议适配与兼容入口
│   ├── observation/  Observation v1 构造与提取
│   └── rules/    单座位规则引擎、候选动作与规则驱动提取器
├── tests/        自动化测试
└── pyproject.toml
```

## 统一事件流

天凤标签只在 `src/suhou/tenhou/` 内解释，之后统一成同步、确定顺序的内部事件：

```text
Parser(TenhouEvent) -> TenhouAdapter -> 不可变 GameEvent
    -> 捕获事件前状态视图 -> ReplayEngine.apply() -> 可见历史/审计输出
```

- `TenhouEvent`：原始 XML 子节点、属性与顺序，仅用于解析、适配和原始审计；旧名 `Event` 仍可用。
- `GameEvent`：已经发生的麻将行为，带 `schema_version=1`、`source_index`、`round_id`、局内 `seq`、`actor` 和按事件种类区分的 payload。牌保留实体编号。
- 事件种类：`metadata`、`round_start`、`draw`、`discard`、`call`、`riichi_declaration`、`riichi_accepted`、`dora`、`result`、`connection`。
- `round_id` 是第几次 `INIT`（从 0 开始），连庄时与 `seed[0]` 相同但 `round_id` 不同。`seq` 是局内源事件序号，`INIT` 为 0；局内元数据不占 `seq`，因此序号可以有间隔。
- `ReplayEngine` 是唯一修改牌局状态的模块，不依赖 XML 标签、属性字典或副露位编码。适配器只读取不可变的 `RoundView` 计算上下文（摸切、岭上摸牌、立直宣言牌、加杠对应的原碰）。
- 连续和了记录共享 `result_group_id`，最后一条标记结算组结束；分数按源顺序逐条应用，整局只关闭一次。结果相关观测使用结算组开始前冻结的状态和历史截止位置。
- `project_visible(event, viewer, red_fives=...)` 采用字段白名单投影。他家摸牌没有牌张、起手事件只有观察者自己的手牌、和了手牌/役种/里宝牌不进入可见历史。赤五标记按 `<GO type>` 规则位解释。
- `iter_replay(source)` 按源顺序产出不可变的 `ReplayFrame`（事件、前后状态视图、消息）；`replay_events(events)` 直接重放内部事件流；`read_event_stream`/`write_event_stream` 读写 `events.jsonl`。

```python
from suhou.replay import read_event_stream, replay_events
from suhou.tenhou import iter_replay

for frame in iter_replay("data/raw/tenhou/sample.mjlog"):
    print(frame.event.kind, frame.message)

outcome = replay_events(read_event_stream("artifacts/tenhou/sample/events.jsonl"))
print(outcome.final_scores)
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
- `events.jsonl`：带 `schema_version=1` 的统一内部事件，一条源事件一行，可用 `read_event_stream()` 重新载入并交给 `replay_events()` 确定性重放。

`replay_tenhou()` 是统一引擎的兼容包装，保留原有 `ReplayResult` 接口、逐源事件的 `replay.jsonl` 审计和 `before_event` 源事件回调（回调拿到的是不可变 `RoundView`）；新代码优先使用 `iter_replay()`。

`--step N` 仅将终端打印限制到第 N 个事件（从 0 开始）；文件始终覆盖整场牌局。程序验证实体牌副本、行动者、副露来源、立直两阶段、和了手牌/副露/宝牌、各局分数连续性及最终 `owari`，并支持四人牌谱中的吃、碰、明杠、暗杠、加杠、和了和流局记录。状态快照同时给出每家舍牌、手牌、副露、立直阶段、牌墙余数和供托。

风位在快照中以结构化字段给出：`round.round_wind` 是场风枚举（`东/南/西/北`，由 `seed[0] // 4` 解出），`round.players[].seat_wind` 是自风枚举（按 `(seat - dealer + 4) % 4` 相对庄家解出，庄家为东）。两者为 JSON 字符串，`round.name`（如「东1局」）仍保留便于对照。

终局为流局且供托仍有剩余时，按天凤规则将剩余立直棒结算给当时的第一名（并列时按起家顺序），每根 1000 点。该笔只出现在 `owari`、不包含在终局的 `sc` 增减分中；重放会在校验前补上这笔，并在 `replay.jsonl` 的 `terminal_riichi_sticks` 字段中记录归属。

内部点数统一为“点”；XML 的 `ten`、`sc` 是 100 点单位，`owari` 的场分是独立的小数值。程序忠实重放牌谱并展示记录中的役种，不自行计算役种/和了点数。天凤官方读谱仅用于验收时的正确性对照，不属于程序交付功能或运行依赖。

重放状态中的舍牌和副露额外保留内部字段：舍牌带 `tile_id`/`seq`，副露带 `formed_seq`/`updated_seq`/`added_tile`。加杠的 `called_tile` 是原来碰入的牌，`added_tile` 是加杠追加的牌，二者不再混为同一个字段。这些字段用于观察构造和牌张流转检查，不进入 observation。每局快照另带 `round_id`（连庄可区分）；结果记录带 `group_id`（同一次多家和了共享）。和了/流局结果不再重复保存整份原始属性，原始属性始终在每行的 `event.attributes` 中通过 `source_index` 追溯。

本地 `tests/` 目录已加入 `.gitignore`，仅用于开发验证，不纳入版本控制。

## Observation v1

`src/suhou/observation/` 提供结构化的玩家可见局面。observation 的构造器只有一个入口 `build_observation`；逐决策提取可使用包装它的 `iter_decision_points`：

```python
from suhou.observation import RulesSnapshot, iter_decision_points

def rules_state(round_state, event, viewer):
    # 兼容提取器的显式占位。真实振听请使用 suhou.rules.PlayerRuleEngine，
    # 或直接用 suhou.rules.iter_player_decisions。
    return RulesSnapshot((False, False, False, False))

points = list(iter_decision_points(
    "data/raw/tenhou/2023041911gm-0001-0000-cba4cded.mjlog",
    rules_state,
))
observation = points[0].observation
print(observation.to_dict())
```

- observation 包含 `viewer`、`table`、`self`、`players`、`context`、`history`。`players` 始终按绝对座位 0..3 排列；`self` 只给出当前观察者的手牌、摸入牌和振听。
- 牌张使用 `Tile(kind, red)` 表示，赤五保留为 `5m/5p/5s + red=True`，不包含实体牌编号；无赤牌规则的对局按 `<GO type>` 解释为 `red=False`。手牌按牌种、赤牌标记排序（同牌种赤牌在前）；舍牌、副露和历史按发生顺序排列。
- `build_observation(replay_state, rules_state, visible_history, viewer, decision_context)` 逐字段读取当前状态（`RoundState` 或不可变 `RoundView`），返回不可变快照。它不读取对手手牌内容、对手摸入牌、里宝牌或结算内部字段。
- `history` 是本局开始到决策边界的可见事件前缀，由 `project_visible()` 从统一事件白名单投影得到。自己的起手牌和摸牌有实际牌张，对手摸牌只有 `tile=None`；对手的起手牌不会出现在历史里。后续鸣牌只追加新事件，不回写旧样本。兼容提取器在事件执行前取数（不含实际动作），规则驱动路径在事件执行后取数（包含当前触发事件）。
- `seq` 是局内源事件序号，每局从 `INIT` 的 0 重新开始；局内元数据不占序号，因此舍牌和副露的 `seq` 可以有间隔，但始终严格递增。`trigger_seq` 是触发本次决策的摸牌/舍牌/鸣牌事件。
- `table.match_type` 由天凤适配器从 `<GO type>` 规则位一次性解出并保存在状态中；`0x08` 为半庄，`0x10` 为三麻（Observation v1 拒绝三麻）。当前样例 `type=1` 是东风战；它在东4局无人达到 30000 点后南入到南1局，因此 `match_type` 为 `tonpu`，不是半庄。
- `DecisionPhase` 在兼容提取器中仍是按实际记录动作给出的 `discard`/`response`/`riichi`/`self_kan`/`tsumo`；规则驱动路径新增 `turn`/`after_call`/`riichi_discard`/`kan_response`，描述决策窗口而不是已经发生的实际选择。合法动作集合、动作掩码和实际动作标签不放进 observation，仍留给 sample 层。
- `RulesState` 只要求 `furiten(viewer)`。`PlayerRuleEngine` 现在直接实现该协议，因此 `build_observation()` 可以消费真实振听；`RulesSnapshot` 没有默认值，仍保留给旧调用方和测试作为显式适配器。新的 rules factory 接收 `(RoundView, GameEvent, viewer)`；仍检查天凤源事件的旧 factory 需用 `legacy_rules_state_factory()` 显式包装。
- `iter_decision_points` 是逐实际动作截取 observation 的兼容入口，内部使用 `iter_replay()` 的不可变帧，在事件执行前取数。它不会为未记录的“跳过鸣牌/跳过荣和”补样本；立直宣言与随后的舍牌是两个源事件，会各自产生一个 decision point，sample 层可按需合并。多条和了共享一个 decision 状态：所有结果观测都使用结算组开始前的状态与历史截止位置。
- 规则驱动路径使用 `iter_player_decisions(source, viewer)`：它复用 `iter_replay()`，在每个事件后用 `PlayerRuleEngine.update()` 生成候选，再用包含当前触发事件的可见历史构造 observation。候选动作放在 `RuleDecision` 中，不进入 observation 契约。
- `DecisionPoint` 额外带 `round_ordinal`（局实例序号，连庄时同一 `seed[0]` 仍可区分）、`event_seq`（历史截止位置）和源事件标识。这些是审计/落盘字段，不属于 observation 契约。
- `EventType`、`Tile`、`MatchType`、`MeldType` 等通用类型定义在 `suhou.domain`，旧的 `suhou.observation` 导入路径继续可用。

## 规则引擎

`src/suhou/rules/` 为指定的训练座位 `viewer` 提供事件后规则判断。`ReplayEngine` 仍是唯一的状态写入者；规则引擎只读取不可变的 `RoundView`，维护自己的振听、一发、杠与等待等派生状态，并为 `viewer` 返回结构化候选动作。

```python
from suhou.rules import PlayerRuleEngine, iter_player_decisions
from suhou.tenhou import iter_replay

engine = PlayerRuleEngine(viewer=0)
for frame in iter_replay("data/raw/tenhou/sample.mjlog"):
    decision = engine.update(frame.event, frame.after)
    if decision is not None:
        print(decision.kind, decision.actions)

# 或者直接取得“候选 + observation + 源事件审计”：
for point in iter_player_decisions("data/raw/tenhou/sample.mjlog", viewer=0):
    print(point.decision.kind, point.decision.actions)
    print(point.observation.to_dict())
```

决策窗口：

- `turn`：自己摸牌后。切牌、立直、自摸、暗杠、加杠、九种九牌。
- `after_call`：自己吃/碰后。只开放切牌，并按本次实际鸣牌应用食替限制。
- `riichi_discard`：自己立直宣言后。只开放能保持听牌的宣言切牌。
- `response`：他家舍牌后。只计算 `viewer` 的荣和、吃、碰、明杠和 `pass`；没有候选时不创建“只能 pass”的决策。
- `kan_response`：他家加杠后。只检查抢杠荣和与 `pass`。

动作使用 `Tile`、赤五标记和副露 `formed_seq` 表达，是规则层的规范输出：

```text
DiscardAction / ChiAction / PonAction / MinkanAction / AnkanAction /
KakanAction / RiichiAction / TsumoAction / RonAction / KyushuAction / PassAction
```

规则边界：

- 首版只覆盖四人天凤规则；不支持三麻、多人响应仲裁、头跳或“碰优先于吃”的裁决。每个 `viewer` 独立运行。
- 候选只属于 `viewer`，不读取对手手牌、对手摸入牌或未来事件来生成候选；结果 payload 只用于判定一次已经发生的荣和窗口是否被取用，不参与候选生成。可见性测试会替换对手私有手牌并确认候选与 observation 不变。
- 每个事件重新生成候选；即使只有一个强制动作也返回决策点，是否过滤由上层决定。普通同种实体牌不区分，赤五保留语义。
- 和了合法性检查包含和了形与“有役”，但不计算符数、飜数或点数；宝牌/里宝牌/赤宝牌不计为役。和了形覆盖标准形、七对子、国士，并支持吃碰杠后的副露形。
- 立直要求门清（暗杠仍算门清）、至少 1000 点、牌墙至少剩 4 张且存在至少一种切后听牌；宣言后尚未扣 1000 点，宣言牌被荣和时不会产生立直成立决策。立直后的摸牌只允许摸切，以及不改变听牌的暗杠、自摸；禁止加杠。
- 暗杠/加杠/明杠检查牌墙、杠数、现有碰和手牌；海底不能杠（最后一张舍牌不能明杠），九种九牌只在自己的首次摸牌且此前无人鸣牌时开放。
- 三种振听分别处理：舍牌振听由当前听牌与自己舍牌的交集动态计算；同巡振听在“后续事件证明本次荣和机会未被取用”后才落实，并在自己摸牌时清除；立直振听一旦错过荣和便持续到本局结束。振听禁止荣和，但不禁止自摸。14 张手牌的回合/宣言快照以摸切后（移除摸入牌）的听牌计算 `self.furiten`；吃碰后的 14 张快照没有固定听牌，只保留同巡/立直标志。荣和合法性只在 13 张响应快照上判断，因此不存在“选择切牌前的振听歧义”。
- 规则驱动的 observation 历史包含当前触发事件，不包含未来事件；`RuleDecision` 放在 observation 外部，`Observation.to_dict()` 的键集与旧契约完全一致。
- `match_event_action(event)` 只做记录事件到结构化动作的匹配，供验收和后续 sample 层使用；**不生成 pass 训练标签**。牌谱无法区分主动跳过与被其他行动截断的窗口保持未标注。

本地验收使用现有 442 事件、5 局样例：225 个实际动作（切牌、鸣牌、立直宣言、和了）逐一映射到对应事件后的候选集合，全部匹配且唯一。样例共产生 277 个决策窗口（204 个 `turn`、62 个 `response`、7 个 `after_call`、4 个 `riichi_discard`）；其中 7 个响应窗口被实际吃/碰、3 个被实际荣和、52 个是合法的未取用吃碰杠窗口，未取用的荣和窗口为 0。新决策点总数大于 225，未取用窗口没有 pass 标签。旧 `iter_decision_points()` 的 225 点基线、`events.jsonl` 往返和 observation JSON 契约保持不变。本地 `tests/` 不纳入版本控制，交付验证只运行 `uv run pytest` 与 `uv run ruff check .`。

## 当前约定

- 训练/验证/测试集按完整对局划分，不能按单步决策随机划分。
- 所有模型输入只能使用当时可见的信息，不得泄漏未来摸牌或对手手牌。
- 模型输出必须通过合法动作掩码过滤。
- `data/` 与 `artifacts/` 仅保留说明文件，原始数据和大文件不入库。
