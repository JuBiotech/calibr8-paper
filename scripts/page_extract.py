import pathlib
from PyPDF2 import PdfFileReader, PdfFileWriter

# Extract SI from full
input_pdf = PdfFileReader("full.pdf")
pdf_writer = PdfFileWriter()
for n in range(38, 47):
    page = input_pdf.getPage(n)
    pdf_writer.addPage(page)
with pathlib.Path("si.pdf").open(mode="wb") as output_file:
    pdf_writer.write(output_file)

# # Extract all except SI from highlighted version
# input_pdf = PdfFileReader("full.pdf")
# pdf_writer = PdfFileWriter()
# for n in range(0, 38):
#     page = input_pdf.getPage(n)
#     pdf_writer.addPage(page)
# with pathlib.Path("full_without_SI.pdf").open(mode="wb") as output_file:
#     pdf_writer.write(output_file)
