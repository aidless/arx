.PHONY: quickstart demo demo-build demo-test

quickstart:
	python3 examples/02_audit_a_leaderboard.py

demo:
	python3 demo/server.py

demo-build:
	python3 demo/build_static.py

demo-test:
	python3 -m unittest demo/test_demo.py -v
