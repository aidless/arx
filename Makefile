.PHONY: install test eval quickstart demo demo-build demo-test measure repro

install:
	python3 -m pip install -e .

test:
	python3 -m pytest tests -q

quickstart:
	python3 examples/02_audit_a_leaderboard.py

# regenerates docs/measurements/*.json + docs/assets/*.png (simulation, offline)
measure:
	python3 scripts/measure_floor_calibration.py
	python3 scripts/measure_verdict_calibration.py

# one command: everything the README claims, offline, no API key
repro: install test quickstart measure

demo:
	python3 demo/server.py

demo-build:
	python3 demo/build_static.py

demo-test:
	python3 -m unittest demo/test_demo.py -v

# 验收题库：20 条对包契约的用例（eval/cases.jsonl），外加既有 pytest 套件。
# 判据见 README「验收标准与复算」：run_cases 必须 20/20，且本目标退出码为 0。
eval:
	python3 eval/run_cases.py
	python3 -m pytest tests -q
