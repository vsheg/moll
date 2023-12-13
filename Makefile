render:
	quarto render

docs:
	poetry run mkdocs build

notebooks:
	jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb

clean:
	rm -rf .venv .mypy* .pytest* .coverage* coverage.xml htmlcov **/__pycache__

cpu: clean
	poetry install -E cpu

cuda12_local: clean
	poetry install
	poetry run pip install --upgrade "jax[cuda12_local]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

metal: clean
	# This is experimental
	poetry env use 3.11
	poetry install
	poetry run pip install -U jax-metal==0.0.4 jax[cpu]==0.4.11 ml_dtypes==0.2.0 --extra-index-url https://storage.googleapis.com/jax-releases/jax_releases.html

.PHONY: render docs notebooks clean