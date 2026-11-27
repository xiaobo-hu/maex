# VoLoAgent → RoboMemArena 接入配置

本目录支持官方 VoLoAgent 的两条路径，底层连接 RoboMemArena 的 OpenPI 策略：

- `replan`：VLM 子目标规划、视觉监控与重规划，动作全部交给 VLA。
- `replan_grasp`：增加 SAM3 分割与 GraspGen 抓取，放置继续调用 VLA。

用户已选定第二条路径作为下一次试跑配置。上游独立 place 工具只有关节控制路径，
没有 LIBERO 的 7 维 OSC 分支，因此这里不启用 `replan_tools`。
LIBERO 抓取原生使用末端闭环，不经过 cuRobo 关节规划；配置显式选择 `linear`
以免触发无关的 cuRobo 服务预检，不宣称提供 cuRobo 避障能力。
这是 RoboMemArena 适配配置，不是论文 RoboVoLo Full 配置或已完成的基准结果。
当前结果尚不构成与 RoboAgent 的公平配对对照；完整提示、任务信息及预算差异见
[公平性审计](VOLO_FAIRNESS_AUDIT.md)。

## 论文对照

论文批次使用 `baselines/VoLoAgent` 的 Python 3.11 环境。环境依赖可从仓库根目录安装。

```bash
uv venv --python 3.11 baselines/VoLoAgent/.venv
uv pip install --python baselines/VoLoAgent/.venv/bin/python \
  -e baselines/VoLoAgent 'websockets==15.0.1'
```

评测采用原生 `subgoal`、VLM 监控和 `replan`，以双视角 RGB 和共享静态任务原语参考作为输入。
深度、GraspGen 与分割工具在该配置下关闭。DeepSeek `deepseek-flash` 负责初始分解和重规划，
本地 `Qwen/Qwen3.8-27B` 每隔 40 个物理步监控，所有动作由冻结的 `pi05_robomemarena` 执行。
模型路由补丁只应用到该批次的 VoLo 源码快照。

每局上限为 900 秒、2,500 步、Planner 20 次与 Observer 100 次，目标控制频率 20 Hz，
π₀.₅ 每次请求最多执行 20 步。`requests.jsonl` 保存 Planner 与 Observer 的 API 时序及代理后端
真实发出的 VLA 请求。`trace.json` 保存正式计时边界和官方评分。
在仓库根目录运行该批次归档中的 `provenance/run_and_report.sh` 可恢复缺失的 Task/seed，
并在 260 局齐全后重新生成汇总。 [实验口径](../../../paper_result/MAEX_RoboMemArena_实验规划与指标.md)

## 源码与独立依赖

官方仓库：<https://github.com/NVlabs/VoLoAgent>。
当前配置检查使用 commit `b4e6230`。上游源码和虚拟环境位于已忽略的
`benchmark/.external/VoLoAgent/`，不修改项目 `.venv`、仿真环境或 VLA 环境。

初次安装：

```bash
git clone --depth 1 https://github.com/NVlabs/VoLoAgent.git benchmark/.external/VoLoAgent
git -C benchmark/.external/VoLoAgent checkout b4e6230
.venv/bin/python -m venv benchmark/.external/VoLoAgent/.venv
benchmark/.external/VoLoAgent/.venv/bin/python -m pip install -e benchmark/.external/VoLoAgent
# 本地 DeepSeek API 兼容补丁；已应用的工作目录不重复执行。
git -C benchmark/.external/VoLoAgent apply \
  "$PWD/benchmark/robomemarena/baselines/volo_patches/deepseek-chat.patch"
git -C benchmark/.external/VoLoAgent apply \
  "$PWD/benchmark/robomemarena/baselines/volo_patches/task-reference.patch"
```

上游依赖范围较宽；本机安装版本保存于该目录的 `installed-requirements.txt`，
源码 commit 保存于 `UPSTREAM_COMMIT`。环境准备只检查依赖、接口和 CPU 数值逻辑，
不执行 GPU 推理、环境连通性或完整 rollout 验证。

## SAM3 与 GraspGen 独立依赖

抓取服务使用 Docker 镜像 `volo-robomemarena-grasp:sam3`，与代理 `.venv` 和其他
实验环境隔离。基础镜像固定为
`ghcr.io/chicychen/grasp-server@sha256:9e39a8d04f6b04b4d1094de386ec614a3b0d4477a6fad2584343691a84c6f26c`。
构建配方见 [`volo_grasp/Dockerfile`](volo_grasp/Dockerfile)。

所需本地资产：

| 资产 | 路径／来源 |
| --- | --- |
| SAM3 源码 | `benchmark/.external/volo-sam3`，官方 VoLo 配方指定的 `Max-Fu/sam3` fork |
| SAM3 权重 | 复用 `models/harnessvla/sam3/sam3.pt`，只读挂载 |
| Panda GraspGen 配置及生成器、判别器权重 | `models/volo/graspgen/checkpoints/` |
| GraspGen 权重来源 | `adithyamurali/GraspGenModels`，revision `ec1ccbb5eec0680db669246ac312a3636f16ee43` |

`models/volo/assets_manifest.json` 和 `models/volo/graspgen/manifest.json`
记录权重 SHA256、文件大小及源码版本。无需 Molmo2、GroundingDINO、SAM2 服务或 cuRobo。
基础镜像可能预装额外库，但启动命令仅加载 SAM3 和 Panda GraspGen。

本机为 RTX PRO 6000 Blackwell（`sm_120`）。基础 PyTorch 已含该架构，
spconv 的编译信息确认可通过 PTX 兼容；基础镜像中的 torch-scatter 与 PointNet2
扩展原本只面向 `sm_86/sm_89`，构建配方以 `12.0+PTX` 重新编译它们。
这不更换 GraspGen 权重或 PTV3 网络结构。编译与架构检查不能替代实际 GPU 推理验证。

原生 SAM3 loader 忽略本地 checkpoint 参数并固定下载 Hugging Face 权重。
[`sam3-local-checkpoint.patch`](volo_grasp/sam3-local-checkpoint.patch) 仅在构建快照中
补充本地路径支持，不修改规划、分割算法或主目录的 VoLo 源码。
镜像设置离线权重加载，服务启动时不再下载模型。

基础镜像和源码下载完成后，可执行以下命令构建；它不暴露 GPU、不加载模型：

```bash
# 首次准备 SAM3 源码时执行：
git clone https://github.com/Max-Fu/sam3.git benchmark/.external/volo-sam3
git -C benchmark/.external/volo-sam3 checkout 6fe87d64a5beb9084923d7a9e002741178635b09

# 源码及表中权重已就绪后构建：
bash benchmark/robomemarena/scripts/prepare_volo_grasp.sh
```

本机 Docker daemon 的旧代理地址不可达，直接 `docker pull` 可能失败。
准备过程中使用用户态下载取得同一 digest 的镜像，逐层核对 SHA256 后再
`docker load` 为 `volo-grasp-base:9e39a8d04f6b`；没有修改 daemon 全局配置。
镜像与源码版本记录位于 `benchmark/.external/volo-tools/`。
该目录的 `wheels/` 保存构建所需的 Python 3.12 安装包及 torch-scatter 2.1.2 源码包，
`wheels-manifest.json` 保存 SHA256。Docker 构建使用 `--network=none`，避免继承旧代理。
`packaging/` 固定 setuptools 77.0.3、packaging 24.2 和 timm 1.0.17，
兼容 VoLo 的许可证字段、SAM3 的模块导入和视觉骨干依赖。

GraspGen 的包元数据仍固定 Torch 2.1.0、spconv-cu120 等旧环境，而官方基础镜像
使用较新的 PyTorch、spconv-cu124，并省略部分训练／数据生成依赖。
本配置沿用该推理环境，`pip check` 不会全绿；不能据此宣称完整训练依赖已满足。
此处验证的是指定推理模块导入、Blackwell 架构、本地资产和客户端接口。
真实 GPU 推理、首次 PTX 编译及控制效果仍需 GPU 空闲后验证。

## 配置

非密钥默认值见 [`../configs/volo.env`](../configs/volo.env)：

| 项目 | 默认值 |
| --- | --- |
| VoLo 监听端口 | `127.0.0.1:8040` |
| 上游 VLA | `127.0.0.1:8000`，应为 `pi05_robomemarena` |
| 环境客户端 | `127.0.0.1:8001` |
| VLM 监控间隔 | 40 个物理动作步；20 Hz 控制下为每 2 秒仿真时间一次（0.5 Hz） |
| 每次执行的 action chunk | 5 步，沿用上游 LIBERO 客户端默认 |
| 规划与监控模型 | 根目录 `.env` 的 `LLM_MODEL_NAME` |
| 抓取服务 | `127.0.0.1:8043`，由 `GRASP_SERVER_PORT` 覆盖 |
| 原生抓取开关 | `VOLO_GRASP_ENABLED=1`，选择 `replan_grasp + sam3` |
| 外部 Skill／独立 place 工具 | 不启用，放置使用 VLA |

`serve_volo.sh` 只读取密钥文件的 `LLM_API_KEY`、`LLM_BASE_URL`、`LLM_MODEL_NAME`，
分别映射到 `VLM_API_KEY`、`VLM_BASE_URL`、`VLM_MODEL`。
已设置的 `VLM_*` 优先；`VOLO_ENV_FILE` 可指定另一个文件。
密钥通过环境变量传递，不写入命令行或配置快照。
VoLo 原生 monitor 使用同一个 VLM 配置；它不是 RoboAgent 的独立 Observer 服务。
监控通过配置的视觉模型 API 发送图片与提示词，本批使用 DeepSeek API；SAM3 不负责完成判断。
`VOLO_CHECK_INTERVAL` 可覆盖检查步数。默认沿用上游 LIBERO 的 40 步设置，
与论文主实验所述 0.2 Hz 不同；实际调用间隔还受推理、网络和仿真速度影响。
已暂停的 `batch26-seed50-reference-20260913-123952` 同样使用 40 步，其历史配置与结果保持原样。

本地 DeepSeek 补丁仅在模型名为 `deepseek-*` 时，默认发送
`extra_body={"thinking":{"type":"disabled"}}`，保留调用方显式覆盖。
原生 512-token 短响应预算在 DeepSeek 默认思考模式下曾产生空正文与任务分解回退；
禁用思考后保持原预算和规划提示。其他模型请求不变。
参数依据 [DeepSeek 思考模式文档](https://api-docs.deepseek.com/guides/thinking_mode/)。
每次代理启动保存 `upstream.patch` 及其 SHA256，记录实际源码差异。
客户端主动中断会记录 `stop_reason=interrupted`、`valid_rollout=false`。

后续代理还会在每个 episode 的 `vlm_requests/` 保存逐调用 JSON：实际提交 SDK 的
system/user 消息、图片 data URL、模型参数、thinking 配置，以及响应正文、模型标识、
usage、finish reason 和调用时间。请求发出前先落盘；API 异常记录错误类型。
不读取或保存客户端密钥，日志写入失败不会重复 API 请求。此记录功能在暂停批量后新增，
不能补回旧实验未保存的请求；没有因此重新启动任何实验。

用户选择的当前验证为 **Task 1、seed 50、启用 `--task-reference`**。
该开关复用 RoboAgent／HiMe 的 `task_primitive_reference()`，按官方任务描述精确匹配
`agent/roboagent/prompts/fullvlm_v2_26_memory_tasks.json` 中当前任务的
`brief_description`、`primitive_order` 和可选 `notes`，通过统一 VLM 调用入口给
初始分解、视觉监控与重规划提供相同的静态参考。新 episode 会替换或清空该参考。
它不是演化 Skill，也不代表已观察到目标状态。未指定开关时保留无参考路径。
客户端保存 `task_reference.txt` 和内容 SHA256；代理仍自行生成子目标，VLA 只接收当前
子目标文本。与当前 RoboAgent 基础提示相近，要求将 labels 作为步骤边界、顺序和
VLA 措辞的强参考，但实际任务的对象、目标、次数和条件优先于旧标签。

监控完成判断已按用户要求恢复为固定上游版本的原文，包括“目标物体到达目的地”条件；
不再应用 `primitive-completion.patch`。本地 `failure_handlers/vlm.py` 与固定上游版本一致。
没有新增 pick／place／pour 等逐类判据、任务答案或演化 Skill；静态任务参考仍按上文追加。
本机源码已应用这些补丁，当前 13 项 CPU 测试通过，包括监控 BEFORE/NOW 双图传递、
跨回合参考隔离及实际请求记录。已暂停批量使用的是修改后的完成语义，不能重标为原生提示词结果。
恢复原文及新增请求记录功能尚未用于在线 rollout，监控效果问题仍未验证解决。
再次启动代理会自动记录新源码补丁及 SHA256，历史实验不重标为新配置结果。

## GPU 空闲后手动运行

以下命令会启动服务或驱动仿真；环境配置阶段不需要执行。
先按 [RoboMemArena README](../README.md) 启动所需的环境和 VLA 服务，
确认环境端口没有被其他实验使用。一个环境服务不能同时跑两个 episode。
深度接口需要使用更新后的环境服务进程；已经运行的旧服务不会自动加载这次代码。
应在空闲后启动新服务，或在另一空闲端口启动独立环境，并同步修改 `--env-port`。

终端一，在明确空闲的 GPU 启动抓取服务：

```bash
# 将 2 换成当时空闲的 GPU 编号。
VOLO_GRASP_GPU=2 bash benchmark/robomemarena/scripts/serve_volo_grasp.sh
```

若容器能看到 GPU，但报 `libcuda.so.1` 缺失或 `Found no NVIDIA driver`，
检查宿主机驱动库是否被 NVIDIA Container Toolkit 漏挂载。本机首次 GPU 验证遇到
内核与用户态驱动版本不同的情况，以下显式只读挂载已通过 CUDA 张量计算验证，
不修改宿主机驱动或 Docker 全局设置：

```bash
VOLO_GRASP_GPU=2 VOLO_HOST_CUDA_LIB_DIR=/lib/x86_64-linux-gnu \
  bash benchmark/robomemarena/scripts/serve_volo_grasp.sh
```

π 与抓取服务共用 GPU 时，为 π 进程设置 `CUDA_VISIBLE_DEVICES=2` 和
`XLA_PYTHON_CLIENT_PREALLOCATE=false`，避免 JAX 预占大部分显存。
并行验证应使用独立端口，例如 π `8050`、环境 `8051`、抓取 `8053`、VoLo `8054`，
分别通过 policy server 的 `--port`、环境的 `--port`、`GRASP_SERVER_PORT` 和
`VOLO_PORT` 设置，同时将 `VOLO_VLA_PORT` 设为 `8050`，客户端端口同步改为 `8051/8054`。

终端二，启动 VoLo 代理：

```bash
VOLO_GRASP_ENABLED=1 \
VOLO_LOG_DIR=benchmark/robomemarena/logs/volo/task1-seed50-reference-proxy \
  bash benchmark/robomemarena/scripts/serve_volo.sh
```

终端三，运行单任务：

```bash
.venv/bin/python -m benchmark.robomemarena.baselines.volo_eval \
  --task 1 --seed 50 --max-steps 2500 --action-chunk-size 5 --enable-depth --task-reference \
  --env-port 8001 --proxy-port 8040 \
  --run-dir benchmark/robomemarena/logs/volo/task1-seed50-reference
```

每次运行使用新的 `--run-dir`。代理的 `VOLO_LOG_DIR` 保存原生规划与监控日志、
源码版本和启动配置；episode 目录保存 `run_config.json`、`policy_metadata.json`、
`trace.jsonl`、双视角视频、`result.json`、官方 `evaluation.json`、`summary.json`
及 TSV。trace 中的 `orchestrator_episode_log_dir` 关联代理的对应回合。

客户端发送官方任务描述、双相机 RGB、机器人状态、回合 UUID 和物理步数。
`--enable-depth` 额外申请同帧 agentview 深度及相机标定。深度在服务端从
MuJoCo 归一化值转换为米；客户端对抓取 RGB 与深度做相同的垂直翻转，
通过 `image_for_grasp` / `depth_for_grasp` 避免原生 LIBERO 路径额外水平镜像。
VLA 输入保持 RoboMemArena 的原有预处理。默认客户端不申请深度，原行为保持不变。
`--task-reference` 仅追加上述当前任务静态参考，不将 CSR、stage 或 evaluator 状态交给 VoLo。
官方评分在执行结束后单独读取。原生 VoLo 日志用于检查模型错误或回退行为；
客户端的 `valid_rollout` 仅说明执行客户端没有捕获到异常，不能代替这项审计。

正式对比前仍需对齐模型、checkpoint、动作预算、chunk、提示信息和监控频率；
当前 5 步 chunk 与现有某些 RoboAgent/HiMe 配置不同。

## 不启动服务的验证

2026-09-13 本机已完成 GPU 2 验证：π、仿真、SAM3、GraspGen 可共用该卡，
采样到的运行显存约 15 GiB。DeepSeek 图片请求成功，旧请求模型名实际返回
`deepseek-flash`。SAM3／GraspGen 对牛奶盒的独立 GPU 推理返回有效抓取姿态；
Task 1（seed 50，带官方参考）运行 2500 步，CSR 50%、TSR 0。
官方在第 648 步确认饼干入篮，VoLo 始终停留在第一个 pick 子目标；
本回合未触发原生抓取恢复，不能声称已经验证了该恢复路径的物理执行。
配置、评分、源码补丁及视频保存在本地
`benchmark/robomemarena/logs/volo/gpu2-validation-20260913/`。
这些是单回合接入验证结果，不是完整基准成功率。

```bash
benchmark/.external/VoLoAgent/.venv/bin/python -m pip check
benchmark/.external/VoLoAgent/.venv/bin/vlm-orchestrator --help
bash -n benchmark/robomemarena/scripts/serve_volo.sh
.venv/bin/python -m benchmark.robomemarena.baselines.volo_eval --help
PYTHONPATH=benchmark/.external/VoLoAgent .venv/bin/python -m pytest \
  benchmark/robomemarena/tests/test_volo_eval.py \
  benchmark/robomemarena/tests/test_volo_rgbd.py -q

# 不暴露设备、不启动服务、不构造模型；检查编译架构与本地权重参数。
docker run --rm --runtime=runc --network=none \
  -e NVIDIA_VISIBLE_DEVICES=void -e CUDA_VISIBLE_DEVICES= \
  -v "$PWD/models/volo/graspgen:/models/graspgen:ro" \
  -v "$PWD/models/harnessvla/sam3:/models/sam3:ro" \
  -v "$PWD/benchmark/robomemarena/baselines/volo_grasp/check_install.py:/check_install.py:ro" \
  --entrypoint python3 volo-robomemarena-grasp:sam3 /check_install.py
```
