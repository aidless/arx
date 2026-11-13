.PHONY: install test quickstart demo demo-build demo-test measure repro

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
