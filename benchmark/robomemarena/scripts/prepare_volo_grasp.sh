#!/usr/bin/env bash
# 安装层：从已下载源码与基础镜像构建 SAM3/GraspGen；不启动服务或暴露 GPU。
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"
TOOLS=benchmark/.external/volo-tools
CONTEXT=benchmark/.external/volo-grasp-build
mkdir -p "$TOOLS" "$CONTEXT/VoLoAgent" "$CONTEXT/sam3" "$CONTEXT/wheels" "$CONTEXT/packaging"

# 基础镜像固定 digest；已有该镜像时不会重复下载。
BASE_DIGEST=sha256:9e39a8d04f6b04b4d1094de386ec614a3b0d4477a6fad2584343691a84c6f26c
BASE_TAG=volo-grasp-base:9e39a8d04f6b
if ! docker image inspect "$BASE_TAG" >/dev/null 2>&1; then
  docker pull "ghcr.io/chicychen/grasp-server@$BASE_DIGEST"
  docker tag "ghcr.io/chicychen/grasp-server@$BASE_DIGEST" "$BASE_TAG"
fi
git -C benchmark/.external/VoLoAgent archive HEAD | tar -x -C "$CONTEXT/VoLoAgent"
git -C benchmark/.external/volo-sam3 archive HEAD | tar -x -C "$CONTEXT/sam3"
cp benchmark/robomemarena/baselines/volo_grasp/{Dockerfile,sam3-local-checkpoint.patch} "$CONTEXT/"
# 使用预下载的 Python 3.12 wheels 与 torch-scatter 源码，构建无需外网。
cp "$TOOLS"/wheels/* "$CONTEXT/wheels/"
cp "$TOOLS"/packaging/* "$CONTEXT/packaging/"
docker build --network=none -t volo-robomemarena-grasp:sam3 "$CONTEXT"
docker image inspect volo-robomemarena-grasp:sam3 --format '{{.Id}}' > "$TOOLS/grasp-image-id.txt"
git -C benchmark/.external/VoLoAgent rev-parse HEAD > "$TOOLS/volo-revision.txt"
git -C benchmark/.external/volo-sam3 rev-parse HEAD > "$TOOLS/sam3-revision.txt"
