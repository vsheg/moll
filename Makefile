render:
	quarto render

notebooks:
	jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb

clean:
	rm -rf .venv .mypy* .pytest* .coverage* coverage.xml htmlcov **/__pycache__ 

cpu:
	poetry install -E cpu

cuda12_local:
	poetry install
	pip install --upgrade "jax[cuda12_local]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html


.PHONY: render notebooks clean