.PHONY: figs paper bib all
all: figs paper
bib:
	cd refs && python3 build_bib.py
figs:
	cd code && python3 make_all_figs.py
paper: bib
	pdflatex -interaction=nonstopmode main.tex && pdflatex -interaction=nonstopmode main.tex && pdflatex -interaction=nonstopmode main.tex
