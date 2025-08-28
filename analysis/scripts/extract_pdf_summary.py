import sys
from pathlib import Path

try:
    from PyPDF2 import PdfReader
except Exception as e:
    print('MISSING_DEP: PyPDF2 not installed')
    sys.exit(2)

if len(sys.argv) < 2:
    print('Usage: extract_pdf_summary.py path/to/report.pdf')
    sys.exit(1)

p = Path(sys.argv[1])
if not p.exists():
    print(f'File not found: {p}')
    sys.exit(1)

reader = PdfReader(str(p))
num_pages = len(reader.pages)
print(f'File: {p}\nPages: {num_pages}\n')

# Try to extract text snippets from each page
pages_with_text = 0
for i, page in enumerate(reader.pages):
    text = page.extract_text() or ''
    text = text.strip()
    if text:
        pages_with_text += 1
    snippet = text[:400].replace('\n', ' ')
    print(f'--- Page {i+1} (text length: {len(text)}):')
    if snippet:
        print(snippet)
    else:
        print('[no extractable text on this page; likely image/plot]')
    print()

print(f'Summary: {pages_with_text}/{num_pages} pages contained extractable text.')
