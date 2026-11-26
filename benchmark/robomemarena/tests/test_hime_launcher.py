"""用真实本地 socket 和子进程验证 HiMe 连续启动，不加载模型或环境。"""

from __future__ import annotations

import socket
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_hime_eval.sh"


def probe(port):
    # 执行启动脚本本身的检查代码，避免测试与生产使用两套端口判定。
    source = SCRIPT.read_text().split("<<'PY'\n", 1)[1].split("\nPY", 1)[0]
    return subprocess.run(
        [sys.executable, "-", str(port)], input=source, text=True, capture_output=True,
        timeout=5,
    )


def test_port_probe_accepts_time_wait_after_previous_run():
    with socket.socket() as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        with socket.create_connection(("127.0.0.1", port)) as client:
            peer, _ = listener.accept()
            peer.close()
            assert client.recv(1) == b""
    # 服务端先关闭连接，构造前一任务退出后仍存在的 TIME_WAIT。
    for _ in range(3):
        result = probe(port)
        assert result.returncode == 0, result.stderr


def test_port_probe_rejects_an_actual_listener():
    with socket.socket() as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        result = probe(listener.getsockname()[1])
        assert result.returncode != 0
        assert "occupied" in result.stderr


def test_cleanup_waits_for_service_to_finish(tmp_path):
    marker = tmp_path / "closed"
    source = """
import signal, sys, time
from pathlib import Path
def stop(signum, frame):
    time.sleep(0.4)
    Path(sys.argv[1]).write_text('closed')
    sys.exit(0)
signal.signal(signal.SIGTERM, stop)
print('ready', flush=True)
while True:
    time.sleep(1)
"""
    process = subprocess.Popen(
        [sys.executable, "-u", "-c", source, str(marker)],
        stdout=subprocess.PIPE, text=True,
    )
    try:
        assert process.stdout.readline().strip() == "ready"
        functions = SCRIPT.read_text().split("service_pids=()\n", 1)[1].split("embedding_args=()", 1)[0]
        result = subprocess.run(
            ["bash", "-c", "set -euo pipefail\nservice_pids=()\n" + functions + '\nstop_service_tree "$1"',
             "hime-cleanup-test", str(process.pid)],
            capture_output=True, text=True, timeout=5,
        )
        assert result.returncode == 0, result.stderr
        assert marker.read_text() == "closed"
        assert process.wait(timeout=1) == 0
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=5)
        process.stdout.close()
