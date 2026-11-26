#!/usr/bin/env bash
# 使用单独端口启动 RoboMemArena 基线组件，或运行一个基线客户端。
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$PROJECT"
mkdir -p "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR"
case "${1:-}" in
  arena)
    shift
    export CUDA_VISIBLE_DEVICES="${BASELINE_GPU:-2}"
    export MUJOCO_EGL_DEVICE_ID="${BASELINE_EGL_GPU:-$CUDA_VISIBLE_DEVICES}"
    simulator="$(conda info --base)/envs/roboagent-robomemarena/bin/python"
    exec "$simulator" -m benchmark.robomemarena.baselines.harness_server --port "$BASELINE_ENV_PORT" "$@"
    ;;
  vla)
    shift
    export CUDA_VISIBLE_DEVICES="${BASELINE_GPU:-2}" XLA_PYTHON_CLIENT_PREALLOCATE=false
    export OPENPI_DATA_HOME="$BASELINES/.runtime/openpi"
    exec models/openpi/.venv/bin/python "$BASELINES/robomemarena/serve_policy.py" \
      --port "$BASELINE_VLA_PORT" --checkpoint "${BASELINE_CHECKPOINT:-$PROJECT/models/openpi/checkpoints/pi05_robomemarena}" "$@"
    ;;
  qwen)
    shift
    export CUDA_VISIBLE_DEVICES="${BASELINE_GPU:-4,5}" VLLM_USE_FLASHINFER_SAMPLER=0
    export VLLM_CACHE_ROOT="$BASELINES/.runtime/vllm"
    exec models/.venv-observer/bin/vllm serve "$PROJECT/models/Qwen3.8-27B" \
      --host 127.0.0.1 --port "$BASELINE_QWEN_PORT" --served-model-name "$BASELINE_QWEN_MODEL" \
      --tensor-parallel-size 2 --dtype bfloat16 --max-model-len 16384 --max-num-seqs 8 \
      --gpu-memory-utilization 0.90 --reasoning-parser qwen3 \
      --enable-auto-tool-choice --tool-call-parser qwen3_coder \
      --default-chat-template-kwargs '{"enable_thinking":false}' \
      --limit-mm-per-prompt '{"image":12,"video":0}' "$@"
    ;;
  volo-proxy)
    shift
    load_baseline_llm
    export VLM_API_KEY="$LLM_API_KEY"
    exec "$BASELINES/VoLoAgent/.venv/bin/vlm-orchestrator" --env libero --host 127.0.0.1 \
      --port "$BASELINE_VOLO_PORT" --vla-host 127.0.0.1 --vla-port "$BASELINE_VLA_PORT" \
      --mode subgoal --failure-monitor vlm --recovery-mode replan --motion-planner linear \
      --check-interval 40 --vlm-base-url "$LLM_BASE_URL" --vlm-model "$LLM_MODEL_NAME" "$@"
    ;;
  volo)
    shift
    exec .venv/bin/python -m benchmark.robomemarena.baselines.volo_eval \
      --env-port "$BASELINE_ENV_PORT" --proxy-port "$BASELINE_VOLO_PORT" "$@"
    ;;
  volo-grasp)
    shift
    export CUDA_VISIBLE_DEVICES="${BASELINE_GPU:-6}" HF_HUB_OFFLINE=1
    deps="$BASELINES/robomemarena/.deps"
    export PYTHONPATH="$BASELINES/VoLoAgent:$deps/graspgen:$deps/sam3"
    cd "$PROJECT/models/volo/graspgen/checkpoints"
    exec "$BASELINES/VoLoAgent/.venv-grasp/bin/python" -m vlm_orchestrator.grasp.server \
      --host 127.0.0.1 --port "$BASELINE_GRASP_PORT" \
      --gripper-config "$PWD/graspgen_franka_panda.yml" \
      --enable-sam3 --sam3-model "$PROJECT/models/sam3/sam3.pt" "$@"
    ;;
  gamma-agent)
    shift
    unset HTTP_PROXY http_proxy HTTPS_PROXY https_proxy ALL_PROXY all_proxy
    export CUDA_VISIBLE_DEVICES="${BASELINE_GPU:-2}" PORT="$BASELINE_GAMMA_PORT"
    export GAMMA_ROOT="$BASELINES/GAMMA"
    export GAMMA_SAM3_MODEL="$PROJECT/models/sam3"
    export GAMMA_HOST=127.0.0.1
    export GAMMA_WRITER_URL="$BASELINE_QWEN_URL" GAMMA_WRITER_MODEL="$BASELINE_QWEN_MODEL"
    export PROMPTS="$BASELINES/GAMMA/rma/prompts_rma.json"
    export WAM_SETTLE_TICKS=2
    exec "$BASELINES/GAMMA/.venv-agent/bin/python" -u "$BASELINES/GAMMA/rma/rma_agent_server.py" "$@"
    ;;
  gamma)
    shift
    gamma="$BASELINES/GAMMA/rma/eval_stack"
    export CUDA_VISIBLE_DEVICES="${BASELINE_GPU:-2}"
    export MUJOCO_EGL_DEVICE_ID="${BASELINE_EGL_GPU:-$CUDA_VISIBLE_DEVICES}"
    export RMA_AGENT_PORT="$BASELINE_GAMMA_PORT" GAMMA_STATELESS_OPENPI=1
    export RMA_PLAN_LABELS="$gamma/rma_their_labels.json"
    export PYTHONPATH="$ROBOMEMARENA_ROOT/third_party/openpi_minimal/packages/openpi-client/src"
    exec "$gamma/.venv/bin/python" "$gamma/run_rma_gamma_eval.py" \
      --adapter-spec "$gamma/mme_rma_adapter.py:build_adapter" \
      --adapter-kwargs "{\"host\":\"127.0.0.1\",\"port\":$BASELINE_VLA_PORT,\"action_horizon\":10,\"prompt_from_subgoal\":true}" \
      --seed 50 --num-trials-per-task 50 --replan-steps 10 --max-steps 2500 --fail-on-extra-pour "$@"
    ;;
  gamma-sweep)
    shift
    exec "$BASELINES/GAMMA/rma/eval_stack/.venv/bin/python" -u "$BASELINES/robomemarena/gamma_sweep.py" "$@"
    ;;
  sam3)
    shift
    export CUDA_VISIBLE_DEVICES="${BASELINE_GPU:-2}" SAM3_CHECKPOINT_PATH="$PROJECT/models/sam3/sam3.pt"
    export PYTHONPATH="$BASELINES/RPent:$BASELINES/robomemarena/.deps/sam3"
    exec "$BASELINES/RPent/.venv-sam3/bin/python" -m rpent.robots.components.sam3_server \
      --transport http --host 127.0.0.1 --port "$BASELINE_SAM3_PORT" "$@"
    ;;
  rpent-vla)
    shift
    export CUDA_VISIBLE_DEVICES="${BASELINE_GPU:-2}" OPENPI_DATA_HOME="$BASELINES/.runtime/openpi"
    openpi="$BASELINES/robomemarena/.deps/rpent-openpi"
    export PYTHONPATH="$BASELINES/RPent:$openpi/src:$openpi/packages/openpi-client/src"
    exec "$BASELINES/RPent/.venv-pi05/bin/python" -m rpent.robots.components.pi05_vla_server \
      --embodiment libero --model-path "$PROJECT/models/harnessvla/RLinf-Pi05-LIBERO-130-fullshot-SFT" \
      --transport http --host 127.0.0.1 --port "$BASELINE_RPENT_VLA_PORT" "$@"
    ;;
  rpent)
    shift
    load_baseline_llm
    planner="${BASELINE_RPENT_PLANNER:-api}"
    model="openai-chat:$LLM_MODEL_NAME"
    if [[ "$planner" == claude_code ]]; then
      endpoint="${LLM_BASE_URL%/}"
      export ANTHROPIC_AUTH_TOKEN="$LLM_API_KEY"
      export CLAUDE_CONFIG_DIR="$BASELINES/.runtime/claude-code"
      if [[ "$endpoint" == https://api.deepseek.com || "$endpoint" == https://api.deepseek.com/v1 ]]; then
        export ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic
      else
        export ANTHROPIC_BASE_URL="${ANTHROPIC_BASE_URL:-${endpoint%/v1}}"
      fi
      model="$LLM_MODEL_NAME"
    else
      export OPENAI_API_KEY="$LLM_API_KEY" OPENAI_BASE_URL="$LLM_BASE_URL"
    fi
    export PYTHONPATH="$PROJECT:$PROJECT/agent:$PROJECT/agent/agent-core"
    exec "$BASELINES/RPent/.venv/bin/python" -m benchmark.robomemarena.baselines.harness_official \
      --harness-root "$BASELINES/RPent" --env-port "$BASELINE_ENV_PORT" \
      --policy-port "$BASELINE_RPENT_VLA_PORT" --sam3-endpoint "http://127.0.0.1:$BASELINE_SAM3_PORT" \
      --planner "$planner" --model "$model" "$@"
    ;;
  rea)
    shift
    if [[ "${REA_RMA_POLICY:-vla}" == chat ]]; then
      load_baseline_llm
      export REA_API_KEY="$LLM_API_KEY" REA_BASE_URL="$LLM_BASE_URL" REA_MODEL="$LLM_MODEL_NAME"
    fi
    exec "$BASELINES/REA/runtime/node/bin/node" "$BASELINES/REA/scripts/rea.mjs" \
      run "$BASELINES/robomemarena/rea.config.mjs" "$@"
    ;;
  *)
    printf '%s\n' 'Usage: run.sh {arena|vla|qwen|volo-proxy|volo|volo-grasp|gamma-agent|gamma|gamma-sweep|sam3|rpent-vla|rpent|rea} [args...]' >&2
    exit 2
    ;;
esac
