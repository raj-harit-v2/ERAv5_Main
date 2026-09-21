"""Load Assign_0013 .env placeholders without python-dotenv."""

from __future__ import annotations

from pathlib import Path
from typing import Dict


def load_env_file(path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not path.is_file():
        return out
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        out[key.strip()] = val.strip()
    return out


def env_float(env: Dict[str, str], key: str, default: float) -> float:
    raw = env.get(key, "")
    if raw == "":
        return default
    return float(raw)


def env_int(env: Dict[str, str], key: str, default: int) -> int:
    raw = env.get(key, "")
    if raw == "":
        return default
    return int(float(raw))


def env_str(env: Dict[str, str], key: str, default: str) -> str:
    raw = env.get(key, "")
    return default if raw == "" else raw
