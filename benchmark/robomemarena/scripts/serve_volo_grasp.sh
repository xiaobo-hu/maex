#!/usr/bin/env bash
# 服务层：手动启动时才绑定指定 GPU；只加载 SAM3 与 Panda GraspGen。
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"
source benchmark/robomemarena/configs/volo.env
: "${VOLO_GRASP_GPU:?Set VOLO_GRASP_GPU to an idle GPU index before starting}"
# 宿主驱动库与内核版本不一致时，容器工具可能漏挂载计算库；仅显式指定时补齐。
driver_mounts=()
if [[ -n "${VOLO_HOST_CUDA_LIB_DIR:-}" ]]; then
  for library in libcuda.so.1 libnvidia-ptxjitcompiler.so.1 libnvidia-nvvm.so.4; do
    driver_mounts+=(-v "$VOLO_HOST_CUDA_LIB_DIR/$library:/usr/local/nvidia/lib64/$library:ro")
  done
fi
exec docker run --rm --name "volo-grasp-$GRASP_SERVER_PORT" \
  --gpus "device=$VOLO_GRASP_GPU" --ipc=private --shm-size=2g \
  "${driver_mounts[@]}" \
  -e "NVIDIA_VISIBLE_DEVICES=$VOLO_GRASP_GPU" -e CUDA_VISIBLE_DEVICES=0 -e HF_HUB_OFFLINE=1 \
  -p "127.0.0.1:$GRASP_SERVER_PORT:8003" \
  -v "$PROJECT_ROOT/models/volo/graspgen:/models/graspgen:ro" \
  -v "$PROJECT_ROOT/models/harnessvla/sam3:/models/sam3:ro" \
  volo-robomemarena-grasp:sam3 \
  --host 0.0.0.0 --port 8003 \
  --gripper-config /models/graspgen/checkpoints/graspgen_franka_panda.yml \
  --enable-sam3 --sam3-model /models/sam3/sam3.pt
