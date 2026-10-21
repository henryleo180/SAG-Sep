PYTHON ?= python

.PHONY: help test-python smoke-train smoke-train-licensed smoke-onnx

help:
	@echo "test-python           run the python test suite"
	@echo "smoke-train           train the packaged sample and export it (no solver, no ONNX)"
	@echo "smoke-train-licensed  the above plus the Gurobi pipeline (generate/label/pack/validate)"
	@echo "smoke-onnx            the above plus the ONNX export, its parity gates and the C engine"
	@echo ""
	@echo "The C engine is built separately: make -C package/c GUROBI_HOME=... ONNXRUNTIME_HOME=... all"

test-python:
	cd package/python && PYTHONPATH=. $(PYTHON) -m pytest tests -q

smoke-train:
	cd package/python && PYTHONPATH=. $(PYTHON) -m sagsep_train smoke --workdir ../../results/smoke-base

smoke-train-licensed:
	cd package/python && PYTHONPATH=. $(PYTHON) -m sagsep_train smoke --licensed --workdir ../../results/smoke-licensed

smoke-onnx:
	cd package/python && PYTHONPATH=. $(PYTHON) -m sagsep_train smoke --onnx --workdir ../../results/smoke-onnx
