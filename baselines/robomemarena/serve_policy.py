"""基线服务层：用 RoboMemArena 的 OpenPI 配置启动本地 VLA 服务。"""

import argparse
import logging
import os
import sys
from pathlib import Path


def main() -> None:
    """从本地 checkpoint 加载 π0.5，将服务绑定至回环地址。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    packages = Path(os.environ["ROBOMEMARENA_ROOT"]) / "third_party/openpi_minimal/packages"
    sys.path[:0] = [str(packages / name / "src") for name in ("openpi", "openpi-client")]

    from openpi.policies.policy_config import create_trained_policy
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    from openpi.training.config import get_config

    logging.basicConfig(level=logging.INFO)
    checkpoint = Path(args.checkpoint).resolve(strict=True)
    policy = create_trained_policy(get_config("pi05_robomemarena"), str(checkpoint))
    WebsocketPolicyServer(policy, host="127.0.0.1", port=args.port, metadata=policy.metadata).serve_forever()


if __name__ == "__main__":
    main()
