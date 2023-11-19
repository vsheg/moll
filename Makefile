render:
	quarto render

notebooks:
	jupyter nbconvert --to notebook --execute --inplace notebooks/d*.ipynb

.PHONY: render notebooks