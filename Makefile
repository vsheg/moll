prep_docs:
	rsync -av notebooks/ docs/notebooks/
	rsync -av README.md docs/index.md

docs: prep_docs
	poetry run mkdocs build

serve: prep_docs
	poetry run mkdocs serve

notebooks:
	jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb

clean:
	rm -rf .venv .mypy* .pytest* .coverage* coverage.xml htmlcov **/__pycache__

metal: clean
	# This is experimental
	poetry env use 3.11
	poetry install
	poetry run pip install -U jax-metal==0.0.4 jax[cpu]==0.4.11 ml_dtypes==0.2.0 --extra-index-url https://storage.googleapis.com/jax-releases/jax_releases.html

.PHONY: docs notebooks clean