# VoLo 视觉工具独立依赖层；构建时不暴露 GPU，也不加载权重。
FROM volo-grasp-base:9e39a8d04f6b
ENV NVIDIA_VISIBLE_DEVICES=void CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1
ENV TORCH_CUDA_ARCH_LIST="12.0+PTX" MAX_JOBS=4 CUDA_HOME=/usr/local/cuda
COPY wheels /opt/volo-wheels
# 基础 PyTorch 已含 sm_120；这两个扩展原来只有 sm_86/sm_89，离线重编译。
RUN FORCE_CUDA=1 python3 -m pip install --no-cache-dir --force-reinstall \
    --no-index --find-links=/opt/volo-wheels --no-deps --no-build-isolation \
    --no-binary=torch-scatter torch-scatter==2.1.2 && \
    python3 -m pip install --no-cache-dir --force-reinstall --no-deps --no-build-isolation \
    /opt/graspgen/pointnet2_ops
COPY packaging /opt/volo-packaging
# 同时满足 VoLo 的 SPDX license 字段、GraspGen 的 setuptools<78 和 SAM3 的 pkg_resources。
RUN python3 -m pip install --no-index --find-links=/opt/volo-packaging \
    setuptools==77.0.3 packaging==24.2 timm==1.0.17
COPY VoLoAgent /opt/volo-robomemarena
COPY sam3 /opt/volo-sam3
COPY sam3-local-checkpoint.patch /tmp/sam3-local-checkpoint.patch
WORKDIR /opt/volo-robomemarena
RUN git apply /tmp/sam3-local-checkpoint.patch && \
    python3 -m pip install --no-cache-dir --no-build-isolation --no-deps -e /opt/volo-sam3 -e . && \
    python3 -m pip install --no-cache-dir --no-index --find-links=/opt/volo-wheels \
    decord==0.6.0 ftfy==6.1.1 pycocotools==2.0.10
# 仅检查导入；不构造 GraspGenSampler 或 SAM3 模型。
RUN python3 -c "import torch, sam3; from grasp_gen.grasp_server import GraspGenSampler; from vlm_orchestrator.grasp.server import main; from spconv.core_cc.cumm.common import CompileInfo; assert 'sm_120' in torch._C._cuda_getArchFlags(); assert CompileInfo.arch_is_compatible((12,0)); assert CompileInfo.arch_is_compatible_gemm((12,0)); print('SAM3, GraspGen and VoLo imports OK; Blackwell PTX compatibility OK')"
ENTRYPOINT ["python3", "-m", "vlm_orchestrator.grasp.server"]
CMD ["--help"]
