"""Serve one serialized LIBERO/MuJoCo environment from Python 3.8."""

from __future__ import annotations

import argparse
import logging

from benchmark.libero.env_server import LiberoEnvironmentServer


def main() -> None:
    """Parse the endpoint and run the environment server until interrupted."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    LiberoEnvironmentServer(args.host, args.port).serve_forever()


if __name__ == "__main__":
    main()
