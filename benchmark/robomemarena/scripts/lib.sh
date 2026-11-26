#!/usr/bin/env bash
# RoboMemArena 评测共享工具:LLM 环境加载、服务启动、端口等待与单任务执行。
# run_eval.sh(单任务冷启动)与 run_batch_persistent.sh(多任务常驻)共同 source。
# 使用前需已 source 配置文件(TASK_ID/SEED/PLANNER/MODEL/POLICY_*/OBSERVER_* 等)。

load_active_llm_env() {
  local env_file="$PROJECT_ROOT/.env"
  [[ -f "$env_file" ]] || return 0
  while IFS= read -r line || [[ -n "$line" ]]; do
    case "$line" in
      LLM_*=*) export "$line" ;;
    esac
  done < "$env_file"
}

load_commented_llm_profile() {
  local env_file="$PROJECT_ROOT/.env"
  local marker="$1"
  local in_profile=0
  [[ -f "$env_file" ]] || return 0
  while IFS= read -r line || [[ -n "$line" ]]; do
    if [[ "$line" == "# $marker" ]]; then
      in_profile=1
      continue
    fi
    if (( in_profile )) && [[ "$line" == "# LLM_"*=* ]]; then
      export "${line#\# }"
    elif (( in_profile )) && [[ -z "$line" ]]; then
      break
    fi
  done < "$env_file"
}

load_planner_llm_env() {
  load_active_llm_env
  # 激活块默认指向一个 provider；PLANNER 不一致时切换 .env 中注释的同名 provider 块。
  case "$PLANNER" in
    qwen)
      if [[ "${LLM_MODEL_NAME:-}" != qwen* ]]; then
        load_commented_llm_profile "qwen 3.8 max api"
      fi
      ;;
    deepseek)
      if [[ "${LLM_MODEL_NAME:-}" != deepseek* ]]; then
        load_commented_llm_profile "deepseek api"
      fi
      ;;
  esac
  export LLM_MODEL_NAME="$MODEL"
  : "${LLM_API_KEY:?Export LLM_API_KEY or configure the matching provider profile in .env}"
  if [[ -n "${ROBOAGENT_HTTP_PROXY:-}" ]]; then
    export HTTP_PROXY="$ROBOAGENT_HTTP_PROXY" HTTPS_PROXY="$ROBOAGENT_HTTP_PROXY"
    export http_proxy="$ROBOAGENT_HTTP_PROXY" https_proxy="$ROBOAGENT_HTTP_PROXY"
    unset ALL_PROXY all_proxy
  fi
  export NO_PROXY="${NO_PROXY:+$NO_PROXY,}127.0.0.1,localhost"
  export no_proxy="${no_proxy:+$no_proxy,}127.0.0.1,localhost"
}

wait_for_service() {
  local port="$1"
  local name="$2"
  local protocol="$3"
  local deadline=$((SECONDS + 180))
  # 32B 权重加载及预热可能超过三分钟，Observer 冷启动留出十五分钟。
  if [[ "$name" == "observer" ]]; then
    deadline=$((SECONDS + 900))
  fi
  until if [[ "$protocol" == "websocket" ]]; then
    .venv/bin/python -c "from websockets.sync.client import connect; socket = connect('ws://127.0.0.1:$port', compression=None); socket.close()" >/dev/null 2>&1
  else
    curl --noproxy '*' --connect-timeout 2 --max-time 5 --fail --silent "http://127.0.0.1:$port/health" >/dev/null 2>&1
  fi; do
    if (( SECONDS >= deadline )); then
      echo "Timed out waiting for $name on port $port" >&2
      return 1
    fi
    sleep 1
  done
}

start_policy_service() {
  local log_dir="$1"
  CUDA_VISIBLE_DEVICES="$POLICY_GPU" \
    models/openpi/.venv/bin/python \
    benchmark/.external/RoboMemArena/third_party/openpi_minimal/scripts/serve_policy.py \
    --port "$POLICY_PORT" \
    policy:checkpoint \
    --policy.config=pi05_robomemarena \
    --policy.dir="$PROJECT_ROOT/models/openpi/checkpoints/pi05_robomemarena" \
    >"$log_dir/policy.log" 2>&1 &
  echo $!
}

start_environment_service() {
  local log_dir="$1"
  (
    conda_base="$(conda info --base)" || exit
    source "$conda_base/etc/profile.d/conda.sh" || exit
    conda activate roboagent-robomemarena || exit
    export CUDA_VISIBLE_DEVICES="$ENV_GPU"
    export ROBOMEMARENA_ROOT="$PROJECT_ROOT/benchmark/.external/RoboMemArena"
    export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
    export NUMBA_CACHE_DIR=/tmp/roboagent-robomemarena-numba
    export MPLCONFIGDIR=/tmp/roboagent-robomemarena-matplotlib
    # 使用服务进程本身的 PID，结束批次时可以直接回收端口。
    exec "$CONDA_PREFIX/bin/python" -m benchmark.robomemarena.scripts.serve_env \
      --host 127.0.0.1 --port "$ENV_PORT"
  ) >"$log_dir/environment.log" 2>&1 &
  echo $!
}

start_observer_service() {
  local log_dir="$1"
  CUDA_VISIBLE_DEVICES="$OBSERVER_GPU" \
    "${OBSERVER_VLLM_BIN:-.venv/bin/vllm}" serve "$PROJECT_ROOT/$OBSERVER_MODEL_PATH" \
    --served-model-name "$OBSERVER_SERVED_MODEL_NAME" \
    --host 127.0.0.1 --port "$OBSERVER_PORT" \
    --dtype bfloat16 --max-model-len 16384 \
    --gpu-memory-utilization "$OBSERVER_GPU_MEMORY_UTILIZATION" \
    "${OBSERVER_VLLM_ARGS[@]}" \
    >"$log_dir/observer.log" 2>&1 &
  echo $!
}

# 统一真实评测的控制回传粒度，避免不同配置文件悄然恢复为较长 action chunk。
run_roboagent_eval() {
  local run_dir="$1"
  ACTION_CHUNK_SIZE=20
  observer_record_args=()
  if [[ "${SAVE_OBSERVER_WINDOWS:-0}" == "1" ]]; then
    observer_record_args=(--observer-record-dir "$run_dir/observer")
  fi
  skill_args=()
  if [[ -n "${TASK_EXECUTION_SKILL:-}" ]]; then
    skill_args=(--task-execution-skill "$TASK_EXECUTION_SKILL")
  fi
  provenance_args=()
  if [[ -n "${RUNTIME_REVISION:-}${EVALUATOR_REVISION:-}${DATA_ROLE:-}" ]] && \
     [[ -z "${RUNTIME_REVISION:-}" || -z "${EVALUATOR_REVISION:-}" || -z "${DATA_ROLE:-}" ]]; then
    echo "RUNTIME_REVISION, EVALUATOR_REVISION and DATA_ROLE must be set together" >&2
    return 2
  fi
  if [[ -n "${RUNTIME_REVISION:-}" && -n "${EVALUATOR_REVISION:-}" && -n "${DATA_ROLE:-}" ]]; then
    provenance_args=(
      --runtime-revision "$RUNTIME_REVISION"
      --evaluator-revision "$EVALUATOR_REVISION"
      --data-role "$DATA_ROLE"
    )
  fi
  # -u 让 stdout 在 tee 管道里实时可见,而不是攒到任务结束才 flush。
  .venv/bin/python -u -m benchmark.robomemarena.scripts.roboagent_eval \
    --policy-host 127.0.0.1 --policy-port "$POLICY_PORT" \
    --env-host 127.0.0.1 --env-port "$ENV_PORT" \
    --task "$TASK_ID" --seed "$SEED" \
    --planner "$PLANNER" --model "$MODEL" \
    --action-chunk-size "$ACTION_CHUNK_SIZE" \
    --max-vla-chunks-per-call "$MAX_VLA_CHUNKS_PER_CALL" \
    --max-agent-turns "$MAX_AGENT_TURNS" \
    --max-steps "$MAX_STEPS" \
    --planner-retries "$PLANNER_RETRIES" \
    --observer-base-url "http://127.0.0.1:$OBSERVER_PORT/v1" \
    --observer-model "$OBSERVER_SERVED_MODEL_NAME" \
    --observer-timesteps "$OBSERVER_TIMESTEPS" \
    --observer-min-new-observations "$OBSERVER_MIN_NEW_OBSERVATIONS" \
    --observer-gripper-threshold "$OBSERVER_GRIPPER_THRESHOLD" \
    --video-fps "$VIDEO_FPS" \
    --run-dir "$run_dir" \
    "${provenance_args[@]}" \
    "${observer_record_args[@]}" \
    "${skill_args[@]}"
}
