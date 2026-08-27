"""Stop a running Session 9 dist server (`python -m src.pipeline.serve_dist`).

Finds listeners on the serve port (default 8765) and/or Python processes whose
command line contains ``serve_dist``, then terminates them.

Usage (from Assign_009_Init project root)::

    python tests/stop_serve_dist.py
    python tests/stop_serve_dist.py --port 8765
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys

DEFAULT_PORT = 8765


def _pids_listening_on_port(port: int) -> set[int]:
    pids: set[int] = set()
    if sys.platform != "win32":
        try:
            out = subprocess.check_output(
                ["lsof", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"],
                text=True,
                errors="replace",
            )
            for line in out.splitlines():
                line = line.strip()
                if line.isdigit():
                    pids.add(int(line))
        except (OSError, subprocess.CalledProcessError):
            pass
        return pids

    try:
        out = subprocess.check_output(
            ["netstat", "-ano"],
            text=True,
            errors="replace",
        )
    except (OSError, subprocess.CalledProcessError):
        return pids

    needle = f":{port}"
    for line in out.splitlines():
        if "LISTENING" not in line:
            continue
        parts = line.split()
        if len(parts) < 5 or not parts[0].upper().startswith("TCP"):
            continue
        local = parts[1]
        if not (local.endswith(needle) or local.endswith("]" + needle)):
            continue
        try:
            pids.add(int(parts[-1]))
        except ValueError:
            continue
    return pids


def _pids_by_wmic_serve_dist() -> set[int]:
    """Windows: find python PIDs with serve_dist in CommandLine via WMIC."""
    pids: set[int] = set()
    if sys.platform != "win32":
        try:
            out = subprocess.check_output(
                ["ps", "ax", "-o", "pid=,args="],
                text=True,
                errors="replace",
            )
        except (OSError, subprocess.CalledProcessError):
            return pids
        me = os.getpid()
        for line in out.splitlines():
            if "serve_dist" not in line:
                continue
            try:
                pid = int(line.split(None, 1)[0])
            except ValueError:
                continue
            if pid != me:
                pids.add(pid)
        return pids

    # WMIC is lighter than PowerShell for this one-shot query
    try:
        out = subprocess.check_output(
            [
                "wmic",
                "process",
                "where",
                "name='python.exe' or name='pythonw.exe'",
                "get",
                "ProcessId,CommandLine",
                "/FORMAT:CSV",
            ],
            text=True,
            errors="replace",
            timeout=15,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return pids

    me = os.getpid()
    for line in out.splitlines():
        if "serve_dist" not in line:
            continue
        # CSV ends with ...,ProcessId
        parts = [p.strip() for p in line.strip().split(",")]
        if len(parts) < 2:
            continue
        try:
            pid = int(parts[-1])
        except ValueError:
            continue
        if pid != me:
            pids.add(pid)
    return pids


def _kill(pid: int) -> bool:
    if pid <= 0 or pid == os.getpid():
        return False
    try:
        if sys.platform == "win32":
            subprocess.check_call(
                ["taskkill", "/PID", str(pid), "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        else:
            os.kill(pid, signal.SIGTERM)
        return True
    except (OSError, subprocess.CalledProcessError):
        return False


def stop_serve_dist(port: int = DEFAULT_PORT) -> int:
    """Kill serve_dist processes; return number stopped."""
    targets = _pids_by_wmic_serve_dist() | _pids_listening_on_port(port)
    targets.discard(os.getpid())

    if not targets:
        print(f"No serve_dist process found (port {port} idle).")
        return 0

    stopped = 0
    for pid in sorted(targets):
        ok = _kill(pid)
        print(f"PID {pid}: {'stopped' if ok else 'failed'}")
        if ok:
            stopped += 1
    print(f"Stopped {stopped} process(es).")
    return stopped


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stop python -m src.pipeline.serve_dist (and listeners on --port)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"Port to clear (default {DEFAULT_PORT})",
    )
    args = parser.parse_args()
    stop_serve_dist(port=args.port)


if __name__ == "__main__":
    main()
