#!/bin/sh
#
python ./tests/dev_compute_matrix.py
python ./tests/dev_analyze_matrix.py > ./output/analysis.tex
cd output
pdflatex analysis.tex
cd ..

