#!/usr/bin/env python3
"""
Simple Selenium controller: reads a `config.json` with a `platforms` list and
runs `worker_selenium.py <platform>` for each entry.
"""

import json
import subprocess
import time
from pathlib import Path

CONFIG_PATH = Path("config.json")


def load():
    if not CONFIG_PATH.exists():
        raise SystemExit("Missing config.json - create one with {\"platforms\": []}")
    with open(CONFIG_PATH) as f:
        return json.load(f)


def main():
    cfg = load()
    platforms = cfg.get("platforms", [])

    print("=" * 60)
    print("BRAF SELENIUM CONTROLLER")
    print("Engine        : selenium")
    print("Max browsers  : 1")
    print("Platforms     :", platforms)
    print("=" * 60)

    for p in platforms:
        print(f"[RUN] {p}")
        try:
            r = subprocess.run(
                ["python3", "worker_selenium.py", p],
                timeout=90
            )
        except subprocess.TimeoutExpired:
            print(f"Timeout running {p}")
        time.sleep(3)

    print("=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
