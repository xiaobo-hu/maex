# VoLo → RoboMemArena 公平性审计

审计日期：2026-09-13。对象为 `gpu2-validation-20260913/task1-seed50-reference`。

当前适配状态：按用户最新要求，监控完成条件已恢复固定上游原文，包括“目标物体到达目的地”。
同一静态参考仍提供给分解、监控和重规划，新回合会清空旧参考；保留原生恢复触发规则，
不新增逐类 primitive 完成条件。此前暂停批量曾使用修改后的完成语义，不能重标为原生提示词结果。
下表及历史分数描述的是审计对象回合；当前改动不追溯改变历史运行条件，恢复后未启动实验。

**结论：当前结果仅可标为带静态参考的移植接入验证，不能用于宣称 VoLo 弱于 RoboAgent，
也不能称为论文 RoboVoLo Full 的复现结果。尚未建立配置冻结、输入资源对齐的配对对照。**

## 证据范围

本次核对保存的 `run_config.json`、`launch_config.json`、`upstream.patch`、参考文本、
trace、官方评分，以及对应 VoLo 源码。审计当时的 VoLo 差异 SHA256 与该回合保存值一致；
后续适配已改变当前源码，复核旧运行应使用其保存的补丁快照。
完整分解 system 提示、分解 user 文本重建、监控 system 提示及文件哈希保存于本地
`benchmark/robomemarena/logs/volo/gpu2-validation-20260913/fairness-audit/`。

重建文件用占位项表示图片，不是当时逐请求发送的完整 messages 档案。
当前 RoboAgent system／Observer 提示也已复制，但不能将当前工作区源码视为历史
RoboAgent 评测的提示快照。历史 v011 恢复记录仅用于说明已有配置差异。

## 核对结果

| 项目 | 已确认事实 | 公平性判断 |
| --- | --- | --- |
| 高层任务 | VoLo 收到官方 Task 1 指令 | 对齐了任务描述 |
| 静态参考 | 复用 `task_primitive_reference()`；Task 1 条目与官方参考相同 | 初始分解获得与当前 RoboAgent 相同的参考文本 |
| 完整 JSON | 共享函数只输出 brief description、primitive labels、可选 notes；双方都不是注入整个 JSON | 不能说 VoLo 少拿了双方共同使用的初始参考字段；是否增加 task_block／scene_description，需双方统一规定 |
| 参考后续可用性 | VoLo 补丁仅给初始分解；监控拿到整体任务、当前／剩余子目标，重规划使用原任务及计划等上下文 | 不等于参考贯穿整个决策链，需要明确迁移规则 |
| primitive 语义 | 当前 RoboAgent system 明确强调参考步骤边界和当前指令终态；VoLo 默认按 pick-and-place 等原子操作规划，监控强调物体到目的地 | 拆开的 pick/place labels 与监控完成条件存在适配风险；尚未证明是该回合失败的唯一原因 |
| 参考约束强度 | VoLo 收到 advisory 参考后生成 `pick up the cookies` 等改写；RoboAgent system 将 labels 作为 strong reference | 都拿到参考不代表使用约束一致；训练标签保真策略需显式说明 |
| 图像 | 两者共用 RoboMemArena 256×256 双 RGB 预处理；VoLo 发送 JPEG，RoboAgent 观察对象封装 PNG | 没有发现遗漏腕部相机；编码、时序采样与上下文不同，不能声称逐像素／逐窗口一致 |
| 时序和机器人状态 | VoLo 监控主要比较 episode 初始与当前双图；当前 RoboAgent Observer 使用近期窗口，决策还获得 robot state／Reflection | 是方法与可用信息的混合差异；要区分同传感器条件和同模型调用资源，不可简单归因于规划算法 |
| 抓取工具 | 原生 `replan_grasp` 的 system 提示仅在抓错物体时允许 `grasp_tool`，对正确物体的尝试要求 continue | 未触发工具符合提示限制，不证明工具接入失效；仅单独验证了抓取预测，未验证物理恢复 |
| 动作工具范围 | 当前配置为 SAM3＋GraspGen、VLA 放置、无独立 place／cuRobo | 必须命名为适配变体；这是双方讨论后选择的可运行范围，不是论文 Full 配置 |
| 模型 | VoLo 的规划与监控均用 DeepSeek，实际返回 deepseek-flash，关闭 thinking；每次短响应默认 512 token | 与历史 qwen3.8-max＋v011 不可直接比较；某份当前 DeepSeek 对照配置另用 Qwen3-VL-32B Observer，也不是同模型体系 |
| 执行预算 | VoLo seed 50、2500 步、5 步 chunk、40 步监控间隔 | 某份当前 RoboAgent seed 52 配置为 20 步 chunk、每调用最多 20 chunks、80 agent turns；不是配对实验，仅总物理步数相同不足以证明公平 |
| 经验和 Skill | VoLo 未加载演化 Skill；历史恢复记录明确含 v011 | 可以比较完整系统，但须披露训练／经验资源；不能把差异全部称为框架收益 |
| 评分隔离 | VoLo 在线未收到官方 stage／CSR／TSR；旁路审计评分独立保存，没有反馈给 Agent | 应继续保留；静态参考本身带有目标位置示例，需统一标为 reference-assisted 条件 |

对应源码：

- `agent/roboagent/prompts/robomemarena.py`、`system.md`、`observer.md` 与 `runtime.py`。
- `benchmark/robomemarena/preprocessing.py`、`client/observation.py`、`baselines/volo_eval.py`。
- VoLo `strategies/subgoal.py`、`strategies/subgoal_base.py`、`failure_handlers/vlm.py`、`cli.py`。
- `configs/deepseek_final_evaluation_v3_seed52_r4.env` 是配置示例证据，不是已指定的配对运行。
- `logs/batches/active-v011-qwen3.8-max-seed50-r1/recovery/run_configuration.json` 是历史恢复记录。

## 结果的正确表述

Task 1 seed 50：2500 步，TSR 0、CSR 50%。官方在第 648 步确认饼干入篮，
VoLo 始终保留第一个 pick 子目标，未执行到番茄酱步骤。可以确认“未切换子目标”的现象，
不能据此断言 VoLo 算法或 DeepSeek 视觉能力更差；输入分辨率、提示语义和迁移约定等
因素还没有被控制。Task 4 无参考回合和中途修改参数前的中断回合均不是公平对照数据。

## 正式对照前的最小要求

1. 指定一份冻结的 RoboAgent 对照配置，明确比较完整系统还是固定模型／资源下的框架。
2. 双方使用同一官方任务信息包与参考政策。对于原子 primitive 的适用范围、完成条件、
   历史状态与当前观察的区别，给出中立的环境／VLA 契约，完成必要的 VoLo 域适配。
   保留各自的规划和记忆机制，不将 RoboAgent 的方法提示整段替换 VoLo。
3. 明确同任务、同 seed／初态、同 checkpoint、同评分与物理动作预算；模型版本、thinking、
   token、监控采样、chunk、额外工具及经验资源固定或单独报告。公平不等于所有提示逐字相同。
4. 保存每个角色实际请求的完整 system/user/tool 文本、图像产物引用及哈希、模型响应标识、
   usage、finish reason、预算与补丁；从中排除密钥。公开提示记录的缺失范围。
5. 用预先指定的开发任务验证适配与工具，再冻结配置；当前 Task 1 已参与调试，后续改善
   应标为开发结果，不能直接当作未见数据上的泛化收益。正式评估另用预先指定的独立执行。

本次审计不启动新实验，也不据此悄悄修改 VoLo 的监控／恢复算法。
