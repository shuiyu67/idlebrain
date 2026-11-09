.PHONY: help test examples build clean check

help:
	@echo "make test      run the unit test suite"
	@echo "make examples  run both example scripts"
	@echo "make build     build sdist + wheel into dist/"
	@echo "make clean     remove build artefacts and caches"
	@echo "make check     test + examples (what CI runs)"

test:
	python3 -m unittest discover -s tests -v

examples:
	python3 examples/quickstart.py
	python3 examples/custom_dataset.py

build: clean
	python3 -m pip install --quiet build
	python3 -m build

check: test

clean:
	rm -rf build dist *.egg-info .pytest_cache
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
	find . -name '*.pyc' -delete
	rm -f *.npz
