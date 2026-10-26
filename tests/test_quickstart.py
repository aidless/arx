import subprocess
from pathlib import Path


def test_quickstart_runs_offline():
    root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        ["make", "quickstart"],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    assert "verdict  : READABLE" in proc.stdout
    assert "UNDECIDABLE" in proc.stdout
    assert "MARGINAL" in proc.stdout
