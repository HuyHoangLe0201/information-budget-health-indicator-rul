.PHONY: figs paper bib all test install supp
install:
	pip install -e ".[models,test]"
test:
	python3 -m pytest -q tests
all: figs paper
bib:
	cd refs && python3 build_bib.py && python3 build_bib.py ../supp/Benchmark_details.tex bib_supp.tex
figs:
	cd code && python3 make_all_figs.py
paper: bib
	for i in 1 2 3; do (cd supp && pdflatex -interaction=nonstopmode Benchmark_details.tex); pdflatex -interaction=nonstopmode main.tex; done

supp:
	cd supp && pdflatex -interaction=nonstopmode Benchmark_details.tex
