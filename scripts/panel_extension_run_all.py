#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = [
    "scripts/panel_extension_1_structural_breaks.py",
    "scripts/panel_extension_2_twfe.py",
    "scripts/panel_extension_3_event_study.py",
    "scripts/panel_extension_4_spillovers.py",
]


def main() -> None:
    py = ROOT / ".venv/bin/python"
    if not py.exists():
        py = Path(sys.executable)

    for script in SCRIPTS:
        path = ROOT / script
        print(f"\n=== Running {path} ===")
        proc = subprocess.run([str(py), str(path)], cwd=str(ROOT))
        if proc.returncode != 0:
            raise SystemExit(f"Failed: {script} (exit {proc.returncode})")

    print("\nAll panel extension scripts completed successfully.")


if __name__ == "__main__":
    main()
