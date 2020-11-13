from pdflatex import PDFLaTeX

pdfl = PDFLaTeX.from_texfile('main_example.tex')
pdf, log, completed_process = pdfl.create_pdf(keep_pdf_file=True, keep_log_file=True)