render:
	quarto render

notebooks:
	jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb

clean:
	rm -rf .venv .mypy* .pytest* .coverage* coverage.xml htmlcov **/__pycache__ 

.PHONY: render notebooks clean