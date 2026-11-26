# RoboMemArena 基线环境

| 基线 | 上游版本 | 本地论文 | RoboMemArena 配置 |
| --- | --- | --- | --- |
| VoLoAgent | [`b4e6230`](https://github.com/NVlabs/VoLoAgent/tree/b4e6230) | [VoLoAgent](VoLoAgent-2606.07723.pdf) | 原代理分解、监督和重规划，通过共享 π₀.₅ 执行动作，可启用 SAM3 与 GraspGen 抓取恢复 |
| GAMMA | [`4370cdc`](https://github.com/1999Lyd/GAMMA/tree/4370cdc) | [GAMMA v1](GAMMA-2610.12793v1.pdf) | 上游 RoboMemArena 子任务序列与机械核验，本地 SAM3、Qwen writer 和 benchmark 发布的 π₀.₅ |
| RPent | [`24541d6`](https://github.com/RLinf/RPent/tree/24541d6) | [Harness VLA v5](Harness-VLA-2607.08448v5.pdf) | 现有 RoboMemArena 适配使用官方 Toolkit、SAM3 和 RLinf π₀.₅ |
| REA | [`56dcda8`](https://github.com/InfRec-Cardinal/REA/tree/56dcda8) | [REA v1](REA-2610.42053v1.pdf) | 原框架的 `python-robot` 接入，支持整任务 π₀.₅ 或 chat 选择子指令 |

RPent 选用项目现有 Harness VLA 适配的固定提交。REA 环境可供后续路由策略接入，当前仿真动作接口为 `vla_chunk`。REA 论文的 LIBERO-Pro 程序池与 RoboTwin 选择器需要各自论文场景的候选策略。

GAMMA 的公开 Drive 目录目前只列出 RoboMME 压缩包。RoboMemArena 论文配置需要 `GAMMA_checkpoints_rma.zip` 中的 writer LoRA 和 `rma_pi05_sgprompt_79999`。当前配置使用 `Qwen3.8-27B` 直接生成 writer 事件，按作者提供的 `rma_their_labels.json` 将当前子任务交给 RoboMemArena 发布权重。报告结果时应记录这两项模型替换。 [上游权重说明](GAMMA/CHECKPOINTS.md) · [公开 Drive 目录](https://drive.google.com/drive/folders/15UWryg_Q7E0GqcG85X8RqZprk1iSMMhq)

## 本机依赖与模型

- 共用 RoboMemArena 的固定源码和只读 Python 3.8 解释器。VoLo 与 REA 的 RPC 客户端读取项目 `.venv`。共享 VLA 与 Qwen 服务分别读取 `models/openpi/.venv` 和 `models/.venv-observer`，以独立端口启动。
- VoLo 代理在 `VoLoAgent/.venv`，使用 `websockets==15.0.1`。抓取服务在 `VoLoAgent/.venv-grasp`，读取已有 Observer 的 CUDA PyTorch，PointNet2 和 torch-scatter 以 RTX A6000 的 `sm_86` 编译。
- GAMMA 仿真客户端在 `GAMMA/rma/eval_stack/.venv`，writer 和 SAM3 在 `GAMMA/.venv-agent`，后者读取已有 Observer 的 PyTorch 和 Transformers。
- RPent Planner 在 `RPent/.venv`。SAM3 使用 `RPent/.venv-sam3` 和本地 SAM3 fork `c1f6b1e`。VLA 使用 `RPent/.venv-pi05`，加载 `rpent-rlinf==0.3.0`、OpenPI `a560f4d`、Transformers `4.57.6`，复用 Observer 的 CUDA PyTorch 和 OpenPI 的通用依赖。
- REA 使用本地 Node `22.21.1`，通过项目的只读 Python 环境运行 RoboMemArena driver。
- Qwen 权重在 `models/Qwen3.8-27B`，共享 VLA 在 `models/openpi/checkpoints/pi05_robomemarena`。新增权重在 `models/harnessvla/RLinf-Pi05-LIBERO-130-fullshot-SFT`、`models/sam3` 和 `models/volo/graspgen/checkpoints`。
- VoLo、RPent 和 REA chat 从项目 `.env` 读取 `LLM_API_KEY`、`LLM_BASE_URL`、`LLM_MODEL_NAME`。设置 `BASELINE_LLM=qwen` 可改用本地 Qwen。GAMMA writer 默认连接本地 Qwen。

VoLo 使用项目已有的 DeepSeek 兼容和任务参考补丁。抓取服务的本地 SAM3 路径与 CUDA 混合精度设置见 [volo-sam3.patch](robomemarena/volo-sam3.patch)。GAMMA 的本地模型接入见 [gamma.patch](robomemarena/gamma.patch)。

## 启动服务

从 `robo-mem-agent` 根目录在单独终端启动所需服务。示例 GPU 编号可改成空闲卡。环境服务的端口为 `8241`，同一进程一次评测一个 episode。各服务可在 `robomemarena/env.sh` 中找到相应的端口覆盖变量。

```bash
BASELINE_GPU=2 bash baselines/robomemarena/run.sh arena
BASELINE_GPU=2 bash baselines/robomemarena/run.sh vla
BASELINE_GPU=4,5 bash baselines/robomemarena/run.sh qwen
```

### VoLoAgent

使用 `arena`、`vla` 和 VoLo 代理。示例使用本地 Qwen 规划。新目录保存每次试跑的日志和评分。

```bash
BASELINE_LLM=qwen bash baselines/robomemarena/run.sh volo-proxy \
  --log-dir baselines/runs/volo-proxy
```

```bash
bash baselines/robomemarena/run.sh volo --task 1 --seed 50 --max-steps 5 \
  --run-dir baselines/runs/volo-smoke
```

抓取恢复配置使用 RGB-D。代理收到恢复指令时可交由 GraspGen 生成抓取位姿，放置继续由 π₀.₅ 执行。

```bash
BASELINE_GPU=7 bash baselines/robomemarena/run.sh volo-grasp
BASELINE_LLM=qwen bash baselines/robomemarena/run.sh volo-proxy \
  --recovery-mode replan_grasp --grasp-seg-mode sam3 --log-dir baselines/runs/volo-grasp-proxy
```

```bash
bash baselines/robomemarena/run.sh volo --task 1 --seed 50 --max-steps 45 --enable-depth \
  --run-dir baselines/runs/volo-grasp-smoke
```

### GAMMA

使用 `vla`、`qwen` 和 GAMMA agent。GAMMA 的原生评测入口在自己的进程中创建 MuJoCo 环境。

```bash
BASELINE_GPU=7 bash baselines/robomemarena/run.sh gamma-agent
```

```bash
BASELINE_GPU=2 bash baselines/robomemarena/run.sh gamma \
  --task-start 1 --task-end 1 --num-trials-per-task 1 --max-steps 12 \
  --out-root baselines/runs/gamma-smoke
```

完整协议覆盖任务 1–26 和 seed 50–99。`gamma-sweep` 使用独立的 RoboMemArena `d09f63b` 检出、GPU 4–7 和本地模型服务，每次恢复时跳过已完成回合。逐回合数据和按任务汇总保存在 `result/gamma_robomemarena`。原论文专用权重包缺失时，这套配置使用 Qwen3.8-27B 和 RoboMemArena 发布版 π₀.₅。

```bash
bash baselines/robomemarena/run.sh gamma-sweep
```

只评测 seed 100 时，每个任务运行一回合，汇总写入 `result/gamma_robomemarena_seed100`。seed 100 对应演示训练布局。

```bash
bash baselines/robomemarena/run.sh gamma-sweep --seed 100
```

### Harness VLA / RPent

使用 `arena`、RPent SAM3 和官方权重服务。默认 Planner 读取 `.env`。选择 `BASELINE_RPENT_PLANNER=claude_code` 时，DeepSeek 官方地址映射到 Anthropic 兼容端点。

```bash
BASELINE_GPU=6 bash baselines/robomemarena/run.sh sam3
BASELINE_GPU=6 bash baselines/robomemarena/run.sh rpent-vla
```

```bash
bash baselines/robomemarena/run.sh rpent --task 1 --seed 50 --explore \
  --explore-attempts-per-session 1 --max-steps 15 --max-turns 4 \
  --memory-dir baselines/runs/rpent-memory --run-dir baselines/runs/rpent-smoke
```

### REA

使用 `arena` 与 `vla`。`--steps` 计 REA 决策次数，`REA_RMA_MAX_PHYSICAL_STEPS` 计 MuJoCo 动作步。默认 task 1、seed 50，可用 `--task` 与 `--seed` 覆盖。设置 `REA_RMA_POLICY=chat` 启用 chat 选子指令。

```bash
REA_RMA_MAX_PHYSICAL_STEPS=2 bash baselines/robomemarena/run.sh rea --steps 1 \
  --output baselines/runs/rea-smoke
```

## 短程验证

Task 1、seed 50 的检查覆盖了以下运行路径。

- VoLoAgent 的 `replan` 和 `replan_grasp` 各执行 45 步，均在第 40 步收到 Qwen 视觉监督结果。抓取服务另用 RoboMemArena 的 RGB-D 完成 SAM3 分割并返回 GraspGen 的 `4×4` 位姿。
- GAMMA 执行 12 步，通过 SAM3、Qwen writer、机械核验和官方评测入口完成两次子任务检查。
- RPent 的 API Planner 和 Claude Code 均调用了仿真观测工具。独立原语测试用 SAM3 识别饼干盒，执行 1 步夹爪原语和 1 个 5 步 π₀.₅ 动作块。权重参数名称与形状核对通过。
- REA 的固定 VLA、`.env` chat 和本地 Qwen chat 配置各执行 2 步。上游 Python driver 与路由运行器的 11 项测试通过。

这些运行用于验证接口和模型加载。完整任务成功率需要按各自评测协议运行。
