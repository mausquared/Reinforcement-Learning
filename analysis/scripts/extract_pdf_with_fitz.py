import sys
from pathlib import Path

try:
    import fitz  # PyMuPDF
except Exception as e:
    print('MISSING_DEP: PyMuPDF not installed')
    sys.exit(2)

if len(sys.argv) < 2:
    print('Usage: extract_pdf_with_fitz.py path/to/report.pdf [out_images_dir] [max_pages_to_render]')
    sys.exit(1)

pdf_path = Path(sys.argv[1])
out_dir = Path(sys.argv[2]) if len(sys.argv) >= 3 else Path('report_images_v2')
max_pages = int(sys.argv[3]) if len(sys.argv) >= 4 else 4

if not pdf_path.exists():
    print(f'File not found: {pdf_path}')
    sys.exit(1)

out_dir.mkdir(parents=True, exist_ok=True)

doc = fitz.open(str(pdf_path))
num_pages = doc.page_count
print(f'File: {pdf_path}\nPages: {num_pages}\n')

pages_with_text = 0
for i in range(num_pages):
    page = doc.load_page(i)
    text = page.get_text().strip()
    print(f'--- Page {i+1} (text length: {len(text)})')
    if text:
        pages_with_text += 1
        snippet = text[:800].replace('\n', ' ')
        print(snippet)
    else:
        print('[no extractable text on this page; likely image/plot]')

    # Render and save up to max_pages
    if i < max_pages:
        mat = fitz.Matrix(2.0, 2.0)  # zoom for higher resolution
        pix = page.get_pixmap(matrix=mat, alpha=False)
        out_path = out_dir / f'page_{i+1:02d}.png'
        pix.save(str(out_path))
        print(f'-- saved page image: {out_path} (size: {out_path.stat().st_size} bytes)')

print(f'Extractable text pages: {pages_with_text}/{num_pages}')
print(f'Images saved to: {out_dir.resolve()}')
