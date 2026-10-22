.PHONY: demo demo-build demo-test

demo:
	python3 demo/server.py

demo-build:
	python3 demo/build_static.py

demo-test:
	python3 -m unittest demo/test_demo.py -v
