# MAEX 11.26v1

MAEX 在 RoboMemArena 中使用 DeepSeek 规划、本地 Qwen 多帧核验和冻结的 π0.5。安装好本仓库的 RoboMemArena、OpenPI 和 Qwen 视觉核验环境后，从仓库根目录运行。启动脚本读取已有的 `.env` 和 `benchmark/robomemarena/configs/deepseek.env`。

```bash
bash maex/run_eval.sh 2 100
```

两个位置参数分别是 Task ID 和 seed。已有相同端口的 policy、环境和 Qwen 视觉核验服务时，使用 `MAEX_EXTERNAL_SERVICES=1 bash maex/run_eval.sh 2 100` 复用。可在 seed 后附加 `--max-steps`、`--verify-stride` 等 MAEX 参数。

多轮评测可在独立终端运行 `bash maex/serve.sh`，复用本机的 π0.5（8000）、Qwen 视觉核验服务（8002）和仿真（8001）。需要并行 episode 时传入独立的仿真端口，例如 `bash maex/serve.sh 8001 8011 8012`。每个仿真端口同一时间运行一个 episode。用 `MAEX_EXTERNAL_SERVICES=1 MAEX_ENV_PORT=8011 bash maex/run_eval.sh 16 100` 选择仿真端口。策略和视觉核验端口也可分别用 `MAEX_POLICY_PORT` 与 `MAEX_OBSERVER_PORT` 覆盖。

## 执行机制

**Executive（执行中枢）** 统一调度视觉核验和动作交接。`Executive.verify()` 按当前意图选择证据窗口，异步调用只读接口 `Observer.verify()`。Qwen 返回带帧引用的候选判断。Executive 通过确定性的 Evidence Gate 核查哪些结果可以写入 Episodic Memory，再通过 Intent Gate 复核下一意图的进入条件，并在动作边界核对交接回执后提交。Executive 是当前 episode 正式状态和控制授权的唯一写入者。

| 文件 | 职责 |
| --- | --- |
| `models.py` | Planner 生成全局有序目标、异步补全后继并在执行停滞时选择恢复。`Observer.verify()` 提供 Executive 的异步视觉核验接口。 |
| `memory.py` | Working Memory 保留关键帧。Evidence Gate 核查帧引用、执行区间和谓词证据。Episodic Memory 记录事件及抽屉内容等绑定证据。固定版本的程序库实例化成功、恢复与补证分支。 |
| `procedural.py` / `reflector.py` | 重放并保存已执行的计划和条件恢复案例。每个 episode 读取固定版本，意图树将匹配的恢复经验绑定到当前步骤。 |
| `executive.py` | Executive 调度核验、接纳证据和接续相容的计划补丁，并依据 Intent Gate 的进入条件及交接回执提交控制权。 |
| `executor.py` | π0.5 沿用原生图像、状态和 7 维动作。执行器在动作边界回执，派发前检查控制代及末端工作区。Executive 可授权在当前意图内开合夹爪、回撤或调用原环境的归位控制。 |

控制按 RoboMemArena 的 20 Hz 上限派发。实际动作反馈达到 `--verify-stride` 的采样跨度时，Executive 可在动作块执行期间发起视觉核验。下一次常规核验从该动作块的结束步计算采样跨度。Planner 回执和视觉核验结果也会唤醒调度。已授权动作可执行至动作块边界，之后派发的动作受待核验窗口的额外步数上限约束。交接仍需通过当前进入条件和 ACK 核查。等待期间步进式仿真保持在当前状态，等待计入墙钟预算。

“放到上一件物品放过的位置”在执行前从已核验的放置事件中绑定。发生提前放置时，带抓持、释放及目标位置证据的相邻后继可直接入账。官方阶段分数从评测通道在 episode 结束后读取。

抓取和放置的核验使用近期帧。实测夹爪开度连续稳定后发起常规抓取核验。常规放置核验在实测开度达到张开阈值或接近空闭时发起。确认放置需识别同一物体，观察到离爪及到达指定区域，并核对夹爪反馈。交接补证和到达尝试上限时的检查仍会执行。视觉核验使用程序库的核验问题、实际图像和反馈。前瞻 Planner 的场景描述供恢复 Planner 参考。相同证据窗口和核验问题只请求一次，重复时记录 `verification_reused`。静止边界上若仍缺证，先使用可授权的补证和恢复动作。可用路径耗尽时记录具体终止原因。

达到 `--attempt-chunks` 指定的动作块数后，Executive 用最新观测复查当前意图。放置途中发现夹爪空闭时会提前复查。抓持复查可以根据最新腕部反证重新打开抓取前提；连续倾倒之间根据后继所需物体判断是否需要重抓，已经核验的倾倒周期保留。恢复回执进入下一次核验，意图推进仍需相应事实和交接许可。

放置动作停滞时，Planner 可根据当前双视角申请定点释放。Executive 核查双视角引用和当前抓持状态，执行后用新观测核验放置效果。若夹爪遮挡结果且双视角确认回撤路径畅通，可短距离退开，再观察落位。

放置过程中检测到持物关系中断时，Executive 可将同一目标已核验的抓取外观附到放置核验和失抓恢复请求。当前帧用于判断物体身份、离爪与落位，抓取外观提供跨时刻对照。

检查抽屉时，Working Memory 将可见开口和同一时刻的内容引用关联。Episodic Memory 保留每层在关闭前的空或有物记录，所有检查完成后才绑定要重开的抽屉。开门核验结合最新固定相机画面的开门效果与可见开口。重复倾倒中掉落的容器可重新抓取，已核验的倾倒周期保留，后续进入条件参照新的抓持证据。

门和抽屉的核验优先比较同一面板与它自己的固定框架。Executive 跨同一目标的重试保留首次动作前的固定前视图，并把当前核验压缩到最多三个时刻。较早的前视图用于识别面板。结果证据仍引用当前窗口内的实际画面。

放置或倾倒期间的夹爪打开与空闭记录在事件记忆中。若夹爪恢复稳定持物，Executive 先用前一次已核验抓取的外观参考补证当前抓持。仍缺证时，恢复 Planner 会收到交接受阻的原因，可以在双视角证实回撤路径畅通时短距离退开并重新核验。外观参考用于认物，当前抓持的证据引用来自本次图像。先前倾倒的引用和次数留在历史记录中。

恢复 Planner 可查看当前步骤最近四次尝试的起止步数、末端位移、夹爪开度和执行回执。遇到重复停滞时，还会比较这次尝试开始和结束的双视角。操作抽屉时，最新双视角显示张开的夹爪对准指定层把手，可先闭合夹爪。握持稳定且路径可见时可沿水平抽拉方向回撤 5 cm。抽屉打开须有持续持握期间的平面位移和指定层开口的视觉证据。

恢复工具返回的结构字段无效时，Executive 会在原动作边界用剩余 Planner 预算请求一次修正。日志记录字段和错误类型。

初始计划校验失败时，在 Planner 和墙钟预算内根据字段错误继续修正。抓持或关门净空缺证时，Executive 会尝试补证、已核验的回撤或保留夹爪命令的归位，再从新视角复查。关门交接沿用已核验的放置事实，并用最新外部相机画面确认手臂净空。复查时物体遮挡作为未知处理，观察到当前矛盾证据时重新核验。末端越界时使用 RoboAgent 的受保护归位轨迹，归位成功后恢复当前任务。

倾倒对象、目的地和次数与 checkpoint 的编号指令一致时，沿用该指令给 π0.5，意图的物体、目标和次数仍由计划与实际证据约束。

夹爪达到稳定抓持后，抓取核验优先提交当前连续持物段的图像。窗口仍覆盖同一次尝试内的抓取前后时，保留持物段之前的临近帧和最近帧。预算允许时还保留实测末端最高的持物帧，供视觉核验比对物体身份和是否离开支撑面。倾倒的核验窗口保留腕部姿态变化较大的关键帧。视觉核验结果引用倾斜和随后回正的画面，重复倾倒要求互不重叠的事件区间。

若 benchmark 的任务文字只要求放置，意图树可在最后一次放置结束。

意图树核验完成后，Executive 保持末端和夹爪命令，通过逐步仿真动作等待官方终态确认。窗口结束仍缺官方确认时，在剩余预算内复查状态并恢复经当前证据否定的可重做动作，保留已核验的倾倒次数。动作、归位和终态核验共用 2,500 物理步与 900 秒默认预算。需要补证且视觉核验预算用尽，或受保护归位无法恢复工作区时，日志会记录相应的终止原因。

每次运行保存到 `benchmark/robomemarena/logs/maex/`。其中 `trace.json` 记录核验、工作窗口、动作边界、补丁接续与控制代，`evaluation.json` 记录官方评分，`videos/` 提供外部和腕部视频。录像使用和 RoboAgent 相同的 `VIDEO_FPS`（当前为 10 FPS），可以用 `--video-fps` 调整播放速度。录像每个可见的仿真状态保存一帧。模型、程序库版本、预算和帧率可查 `run_config.json`。

## 跨 episode 的程序性记忆

`procedures.json` 提供基础谓词、核验要求和默认分支。为多个 episode 指定同一本 SQLite 经验库，可以在这个基础上积累和复用程序。读库与入库分别由 `--experience-db` 和 `--learn-procedures` 控制。

| 范围 | 入库与复用 |
| --- | --- |
| 任务专用 | 任务编号和指令相同、seed 不同。已成功执行且完整通过证据门的计划供 Planner 参考，抽屉选择和继承目的地等槽位在当前 episode 重新绑定。有验证反馈的恢复案例还按具体步骤、物体和目的地匹配。 |
| 通用 | 从其他任务的已执行恢复中按谓词、抽屉类型、执行模式、实测爪态、当前核验结果和交接障碍匹配。至少两个独立 episode 支持且支持数多于反例时，意图树为当前步骤准备相应的恢复候选。同一 seed 中先结束的任务可以为后续任务提供这些经验。 |

Reflector 将归档中的恢复请求与实际派发、物理反馈以及之后的同目标核验对齐。只保存登记过的恢复动作。失败、未决和中断分别记录。已完成轨迹可重复导入，SQLite 事务按 episode ID 去重。每一局启动时锁定库修订号和兼容的执行器、基础程序库版本。新条目可在之后开跑的 episode 中检索。

检索到的恢复候选进入现有 Planner 请求。Planner 按当前双视角重新判断并引用当前帧，Executive 继续使用 Evidence Gate 和 Intent Gate 核查效果、动作条件和交接。历史事实、帧引用和坐标留在原轨迹中。每个目标在一个 episode 内对同一条恢复程序最多审查一次。日志中的 `procedure_retrieved` 记录检索及来源，`procedure_committed` 记录候选对应的恢复实际取得控制权。库修订号和是否允许入库记录在 `run_config.json`。

~~~bash
db="$PWD/benchmark/robomemarena/logs/maex/procedural-11.26v1.sqlite3"
# seed 50 读取已有经验，结束后把通过重放核验的案例写入库
MAEX_EXTERNAL_SERVICES=1 bash maex/run_eval.sh 2 50 --experience-db "$db" --learn-procedures
# seed 100 复用同任务计划及兼容经验
MAEX_EXTERNAL_SERVICES=1 bash maex/run_eval.sh 2 100 --experience-db "$db"
# 同 seed 的另一个任务也可以读取通用经验
MAEX_EXTERNAL_SERVICES=1 bash maex/run_eval.sh 1 50 --experience-db "$db"
~~~

正式对照可给两个批次指定固定的 `--experience-revision N`，并仅使用开发 episode 构建相应版本。已有轨迹及相邻的 `run_config.json` 也能直接重放入库。

~~~bash
.venv/bin/python -m maex.reflector /path/to/run/trace.json --experience-db "$db"
~~~

需要探索新核验问题时，Reflector 可用 DeepSeek 生成带原始事件引用的候选报告，供独立重放和闭环实验复核。

```bash
PROJECT_ROOT="$PWD"
source benchmark/robomemarena/configs/deepseek.env
source benchmark/robomemarena/scripts/lib.sh
load_planner_llm_env
.venv/bin/python -m maex.reflector /path/to/trace.json --output /path/to/candidates.json
```

## 测试

```bash
.venv/bin/python -m pytest -q maex/tests
```

设计依据见 [v3 方案](异步具身智能体_方案设计_v3.md)、[总体框架](MAEX_f1_框架图.pdf)和[异步时序](MAEX_f2_异步.pdf)。
