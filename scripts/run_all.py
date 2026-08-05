"""Regenerate all main-text and supplemental figures.

Run ``python scripts/validate.py`` first. Each script writes a deterministic
figure into ``figures/``.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

HERE = os.path.dirname(__file__)
SCRIPTS = [
    "fig1_topology.py",
    "fig2_superdecoherence.py",
    "fig3_metrology.py",
    "fig4_robustness.py",
    "fig5_esr_stm.py",
    "figS1_phase_certificate.py",
]

for script in SCRIPTS:
    started = time.time()
    print(f"\n=== {script} ===", flush=True)
    result = subprocess.run([sys.executable, os.path.join(HERE, script)], check=False)
    if result.returncode:
        raise SystemExit(result.returncode)
    print(f"    ({time.time() - started:.1f} s)", flush=True)

print("\nAll figures regenerated.")
