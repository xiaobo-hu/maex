"""基线资源准备入口：固定原版 SAM3 和记忆快照，不修改 HarnessVLA 内容。"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from urllib.request import urlopen

SAM3_REVISION = "3c879f39826c281e95690f02c7821c4de09afae7"
MEMORY_REVISION = "8d34642e420e35ea8a883ddabe52471433920177"
BPE_REVISION = "660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7"
BPE_SHA256 = "924691ac288e54409236115652ad4aa250f48203de50a9e4722a6ecd48d6804a"


def main() -> int:
    """下载固定资源，并生成适配原提示词目录约定的独立记忆副本。"""
    from huggingface_hub import hf_hub_download, snapshot_download

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("models/harnessvla"))
    args = parser.parse_args()
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    status = {}
    try:
        snapshot_download(
            "RLinf/RPent-memory", repo_type="dataset", revision=MEMORY_REVISION,
            allow_patterns=["libero/**"], local_dir=root / "resources",
        )
        source = root / "resources/libero"
        if not (source / "MEMORY.md").is_file():
            raise FileNotFoundError("Original memory snapshot is missing libero/MEMORY.md")
        # 官方数据集使用 libero/MEMORY.md，原提示词读取 libero/memory/MEMORY.md。
        # 只调整外部副本的目录层级，保留所有记忆文件的原始字节和相对链接。
        target = root / "resources-compatible/libero/memory"
        shutil.copytree(source, target, dirs_exist_ok=True)
        hashes = {
            str(path.relative_to(source)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(source.rglob("*")) if path.is_file()
        }
        (root / "memory_checksums.json").write_text(json.dumps(hashes, indent=2) + "\n")
        status["memory"] = {
            "ready": True, "repository": "RLinf/RPent-memory", "revision": MEMORY_REVISION,
            "file_count": len(hashes), "resources_dir": str(target.parent),
        }
    except Exception as error:
        status["memory"] = {"ready": False, "error": f"{type(error).__name__}: {error}"}

    try:
        # PyPI 的 sam3 0.1.4 缺少词表文件，单独保存官方原始资源。
        vocabulary = root / "sam3/bpe_simple_vocab_16e6.txt.gz"
        url = (
            f"https://raw.githubusercontent.com/facebookresearch/sam3/{BPE_REVISION}"
            "/sam3/assets/bpe_simple_vocab_16e6.txt.gz"
        )
        if not vocabulary.is_file():
            with urlopen(url, timeout=60) as response:
                content = response.read()
        else:
            content = vocabulary.read_bytes()
        if hashlib.sha256(content).hexdigest() != BPE_SHA256:
            raise ValueError("SAM3 vocabulary checksum mismatch")
        vocabulary.parent.mkdir(parents=True, exist_ok=True)
        vocabulary.write_bytes(content)
        status["vocabulary"] = {
            "ready": True, "source": url, "sha256": BPE_SHA256,
            "path": str(vocabulary),
        }
    except Exception as error:
        status["vocabulary"] = {"ready": False, "error": f"{type(error).__name__}: {error}"}

    try:
        checkpoint = hf_hub_download(
            "facebook/sam3", "sam3.pt", revision=SAM3_REVISION, local_dir=root / "sam3",
        )
        status["sam3"] = {
            "ready": True, "repository": "facebook/sam3", "revision": SAM3_REVISION,
            "checkpoint": checkpoint, "bytes": Path(checkpoint).stat().st_size,
        }
    except Exception as error:
        status["sam3"] = {"ready": False, "error": f"{type(error).__name__}: {error}"}
    (root / "preparation_status.json").write_text(json.dumps(status, indent=2) + "\n")
    for name, result in status.items():
        print(f"{name}: {'ready' if result['ready'] else result['error']}", flush=True)
    return 0 if all(item["ready"] for item in status.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
