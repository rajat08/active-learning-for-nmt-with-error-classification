


all:
	pdflatex main
	bibtex main 
	pdflatex main
	pdflatex modularBF
	rm -rf *.upb *.upa *.dvi *.synctex.gz *.aux *.bbl *.blg *.fls *.log *.out *.thm *.fdb_latexmk 



report:
	latex_check.sh modularB.tex > report.html

clean:
	rm -rf *.upb *.upa *.dvi *.synctex.gz *.aux *.bbl *.blg *.fls *.log *.out *.thm *.fdb_latexmk 
