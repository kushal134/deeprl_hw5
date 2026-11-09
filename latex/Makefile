LATEX ?= latexmk -pdf -interaction=nonstopmode

all: handout.pdf template.pdf

%.pdf: %.tex hw703.sty
	$(LATEX) $<

clean:
	latexmk -C

.PHONY: all clean
