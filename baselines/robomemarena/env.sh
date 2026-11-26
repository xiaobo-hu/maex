# 由基线启动脚本读取的 RoboMemArena 路径、端口和渲染配置。
BASELINES=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PROJECT=$(dirname "$BASELINES")
export ROBOMEMARENA_ROOT="$PROJECT/benchmark/.external/RoboMemArena"
export RMA_BENCH_ROOT="$ROBOMEMARENA_ROOT/evaluation_benchmark"
export LIBERO_CONFIG_PATH="$PROJECT/benchmark/.external/libero-robomemarena-config"
export PYTHONDONTWRITEBYTECODE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export MUJOCO_GL="${MUJOCO_GL:-egl}" PYOPENGL_PLATFORM="${MUJOCO_GL:-egl}"
egl_root="$PROJECT/benchmark/.external/nvidia-egl/usr"
export __EGL_VENDOR_LIBRARY_FILENAMES="$egl_root/share/glvnd/egl_vendor.d/10_nvidia.json"
export LD_LIBRARY_PATH="$egl_root/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export NUMBA_CACHE_DIR="$BASELINES/.runtime/numba" MPLCONFIGDIR="$BASELINES/.runtime/matplotlib"
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}127.0.0.1,localhost"
export no_proxy="$NO_PROXY"
export BASELINE_ENV_PORT="${BASELINE_ENV_PORT:-8241}"
export BASELINE_VLA_PORT="${BASELINE_VLA_PORT:-8240}"
export BASELINE_QWEN_PORT="${BASELINE_QWEN_PORT:-8242}"
export BASELINE_SAM3_PORT="${BASELINE_SAM3_PORT:-8243}"
export BASELINE_VOLO_PORT="${BASELINE_VOLO_PORT:-8244}"
export BASELINE_GAMMA_PORT="${BASELINE_GAMMA_PORT:-8245}"
export BASELINE_RPENT_VLA_PORT="${BASELINE_RPENT_VLA_PORT:-8246}"
export BASELINE_GRASP_PORT="${BASELINE_GRASP_PORT:-8247}"
export GRASP_SERVER_HOST=127.0.0.1 GRASP_SERVER_PORT="$BASELINE_GRASP_PORT"
export BASELINE_QWEN_MODEL="Qwen/Qwen3.8-27B"
export BASELINE_QWEN_URL="http://127.0.0.1:$BASELINE_QWEN_PORT/v1"

load_baseline_llm() {
  if [[ "${BASELINE_LLM:-api}" == qwen ]]; then
    unset HTTP_PROXY http_proxy HTTPS_PROXY https_proxy ALL_PROXY all_proxy
    export LLM_API_KEY=local LLM_BASE_URL="$BASELINE_QWEN_URL" LLM_MODEL_NAME="$BASELINE_QWEN_MODEL"
    return
  fi
  while IFS= read -r line || [[ -n "$line" ]]; do
    case "$line" in
      LLM_API_KEY=*|LLM_BASE_URL=*|LLM_MODEL_NAME=*)
        local key="${line%%=*}" value="${line#*=}"
        printf -v "$key" '%s' "${value%$'\r'}"
        export "$key"
        ;;
    esac
  done < "$PROJECT/.env"
  : "${LLM_API_KEY:?}" "${LLM_BASE_URL:?}" "${LLM_MODEL_NAME:?}"
}
