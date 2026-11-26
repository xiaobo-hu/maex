"""安装验证层：检查离线权重路径、模块导入与 CUDA 架构，不加载模型或运行服务。"""
# ruff: noqa: E402
# 必须先检查设备屏蔽配置，再导入可能探测 CUDA 的第三方模块。

import json
import os
import subprocess
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
    raise SystemExit("Run this installation check with CUDA_VISIBLE_DEVICES='' and no GPU devices")

import pointnet2_ops._ext
import sam3
import torch
import torch_scatter
from grasp_gen.grasp_server import GraspGenSampler, load_grasp_cfg
from spconv.core_cc.cumm.common import CompileInfo
from vlm_orchestrator.perception.sam3 import SAM3Detector

assert not torch.cuda.is_available()
assert callable(GraspGenSampler)
assert "sm_120" in torch._C._cuda_getArchFlags()
assert CompileInfo.arch_is_compatible((12, 0))
assert CompileInfo.arch_is_compatible_gemm((12, 0))

extensions = [Path(pointnet2_ops._ext.__file__)]
extensions.extend(Path(torch_scatter.__file__).parent / name for name in (
    "_scatter_cuda.so", "_segment_coo_cuda.so", "_segment_csr_cuda.so",
))
assert len(extensions) >= 2
for path in extensions:
    architectures = subprocess.check_output(["cuobjdump", "--list-elf", str(path)], text=True)
    assert "sm_120" in architectures, f"Blackwell cubin missing: {path.name}"

checkpoint = Path("/models/sam3/sam3.pt")
assert checkpoint.is_file()
# 用替身确认本地加载参数，避免在安装检查中构造多 GB 的模型。
with patch.object(sam3, "build_sam3_image_model", return_value=SimpleNamespace(parameters=lambda: [])) as builder:
    SAM3Detector(model_id=str(checkpoint), device="cpu")._ensure_loaded()
    assert builder.call_args.kwargs["load_from_HF"] is False
    assert builder.call_args.kwargs["checkpoint_path"] == str(checkpoint)
    assert Path(builder.call_args.kwargs["bpe_path"]).is_file()

cfg = load_grasp_cfg("/models/graspgen/checkpoints/graspgen_franka_panda.yml")
assert cfg.data.gripper_name == "franka_panda"
weight_metadata = {}
for path in (cfg.eval.checkpoint, cfg.discriminator.checkpoint):
    assert Path(path).is_file()
    # meta 设备只解析形状，不分配权重张量或执行网络；保持原生 torch.load 默认参数。
    archive = torch.load(path, map_location="meta", mmap=True)
    assert isinstance(archive["model"], dict)
    weight_metadata[Path(path).name] = len(archive["model"])
sam_archive = torch.load(checkpoint, map_location="meta", mmap=True, weights_only=True)
assert isinstance(sam_archive, dict)

print(json.dumps({
    "status": "passed", "gpu_exposed": False, "model_inference": False,
    "torch": torch.__version__, "sam3": sam3.__version__,
    "torch_scatter": version("torch-scatter"),
    "blackwell_extensions": [path.name for path in extensions],
    "spconv_ptx_compatible": True, "local_sam3_checkpoint": str(checkpoint),
    "gripper": cfg.data.gripper_name,
    "graspgen_state_dict_entries": weight_metadata, "sam3_checkpoint_readable": True,
}, indent=2))
