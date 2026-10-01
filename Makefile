PY ?= python3.11
VENV := .venv
BIN := $(VENV)/bin
export PYTHONPATH := src

.PHONY: setup sample demo test clean

$(BIN)/python:
	$(PY) -m venv $(VENV)
	$(BIN)/pip install -q -r requirements.txt

setup: $(BIN)/python

sample: setup
	$(BIN)/python -m ipo_model.synthetic

demo: sample            ## regenerate synthetic data, metrics and charts in results/
	$(BIN)/python -m ipo_model.evaluate --raw data/sample/ipo_synthetic.csv

test: setup
	$(BIN)/python -m pytest -q

clean:
	rm -rf models data/processed .pytest_cache
